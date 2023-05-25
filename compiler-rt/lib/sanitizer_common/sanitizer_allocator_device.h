//===-- sanitizer_allocator_device.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#  error This file must be included inside sanitizer_allocator.h
#endif

struct DeviceAllocationInfo;
#if SANITIZER_AMDGPU
// Device memory allocation usually requires additional information, we can put
// all the additional information into a data structure DeviceAllocationInfo.
// This is only a parent structure since different vendors may require
// different allocation info.
typedef enum {
  DAT_UNKNOWN = 0,
  DAT_AMDGPU = 1,
} DeviceAllocationType;

struct DeviceAllocationInfo {
  DeviceAllocationInfo(DeviceAllocationType type = DAT_UNKNOWN) {
    type_ = type;
  }
  DeviceAllocationType type_;
};

struct DevivePointerInfo {
  uptr map_beg;
  uptr map_size;
};

#include "sanitizer_allocator_amdgpu.h"

template <class MapUnmapCallback = NoOpMapUnmapCallback>
class DeviceAllocatorT {
 public:
  using PtrArrayT = DefaultLargeMmapAllocatorPtrArray;
  using DeviceMemFuncs = AmdgpuMemFuncs;

  void Init(bool enable, uptr kMetadataSize) {
    internal_memset(this, 0, sizeof(*this));
    enabled_ = enable;
    if (!enable)
      return;
    kMetadataSize_ = kMetadataSize;
    chunks_ = reinterpret_cast<uptr *>(ptr_array_.Init());
    InitMemFuncs();
  }

  void *Allocate(AllocatorStats *stat, uptr size, uptr alignment,
                 DeviceAllocationInfo *da_info) {
    if (!da_info || !InitMemFuncs())
      return nullptr;

    // Allocate an extra page for Metadata
    if (kMetadataSize_ + (size % page_size_) > page_size_) {
      size += page_size_;
    }
    CHECK(IsPowerOfTwo(alignment));
    uptr map_size = RoundUpMapSize(size);
    if (alignment > page_size_)
      map_size += alignment;
    // Overflow.
    if (map_size < size) {
      Report(
          "WARNING: %s: DeviceAllocator allocation overflow: "
          "0x%zx bytes with 0x%zx alignment requested\n",
          SanitizerToolName, map_size, alignment);
      return nullptr;
    }
    void *ptr = DeviceMemFuncs::Allocate(map_size, alignment, da_info);
    if (!ptr)
      return nullptr;
    uptr map_beg = reinterpret_cast<uptr>(ptr);
    CHECK(IsAligned(map_beg, page_size_));
    MapUnmapCallback().OnMap(map_beg, map_size);
    uptr map_end = map_beg + map_size;
    uptr res = map_beg;
    if (res & (alignment - 1))  // Align.
      res += alignment - (res & (alignment - 1));
    CHECK(IsAligned(res, alignment));
    CHECK(IsAligned(res, page_size_));
    CHECK_GE(res + size, map_beg);
    CHECK_LE(res + size, map_end);
    uptr size_log = MostSignificantSetBitIndex(map_size);
    CHECK_LT(size_log, ARRAY_SIZE(stats.by_size_log));
    {
      SpinMutexLock l(&mutex_);
      ptr_array_.EnsureSpace(n_chunks_);
      uptr idx = n_chunks_++;
      chunks_[idx] = map_beg;
      chunks_sorted_ = false;
      stats.n_allocs++;
      stats.currently_allocated += map_size;
      stats.max_allocated = Max(stats.max_allocated, stats.currently_allocated);
      stats.by_size_log[size_log]++;
      stat->Add(AllocatorStatAllocated, map_size);
      stat->Add(AllocatorStatMapped, map_size);
    }
    return reinterpret_cast<void *>(res);
  }

  void Deallocate(AllocatorStats *stat, void *p) {
    Header header, *h;
    {
      SpinMutexLock l(&mutex_);
      uptr idx, end;
      uptr p_ = reinterpret_cast<uptr>(p);
      EnsureSortedChunks();  // Avoid doing the sort while iterating.
      for (idx = 0; idx < n_chunks_; idx++) {
        if (chunks_[idx] >= p_)
          break;
      }
      CHECK_EQ(chunks_[idx], p_);
      CHECK_LT(idx, n_chunks_);
      h = GetHeader(chunks_[idx], &header);
      CHECK_NE(h, nullptr);
      chunks_[idx] = chunks_[--n_chunks_];
      chunks_sorted_ = false;
      stats.n_frees++;
      stats.currently_allocated -= h->map_size;
      stat->Sub(AllocatorStatAllocated, h->map_size);
      stat->Sub(AllocatorStatMapped, h->map_size);
    }
    MapUnmapCallback().OnUnmap(h->map_beg, h->map_size);
    DeviceMemFuncs::Deallocate(p);
  }

  uptr TotalMemoryUsed() {
    Header header;
    SpinMutexLock l(&mutex_);
    uptr res = 0, beg, end;
    for (uptr i = 0; i < n_chunks_; i++) {
      Header *h = GetHeader(chunks_[i], &header);
      CHECK_NE(h, nullptr);
      res += RoundUpMapSize(h->map_size);
    }
    return res;
  }

  bool PointerIsMine(const void *p) const {
    return GetBlockBegin(p) != nullptr;
  }

  uptr GetActuallyAllocatedSize(void *p) {
    Header header;
    uptr p_ = reinterpret_cast<uptr>(p);
    Header *h = GetHeader(p_, &header);
    return h ? h->map_size : 0;
  }

  void *GetMetaData(const void *p) {
    Header header;
    uptr p_ = reinterpret_cast<uptr>(p);
    Header *h = GetHeader(p_, &header);
    return h ? reinterpret_cast<void *>(h->map_beg + h->map_size -
                                        kMetadataSize_)
             : nullptr;
  }

  void *GetBlockBegin(const void *ptr) const {
    Header header;
    if (!mem_funcs_inited_) return nullptr;
    uptr p = reinterpret_cast<uptr>(ptr);
    SpinMutexLock l(&mutex_);
    uptr nearest_chunk = 0;
    // Cache-friendly linear search.
    for (uptr i = 0; i < n_chunks_; i++) {
      uptr ch = chunks_[i];
      if (p < ch)
        continue;  // p is at left to this chunk, skip it.
      if (p - ch < p - nearest_chunk)
        nearest_chunk = ch;
    }
    if (!nearest_chunk)
      return nullptr;
    if (p != nearest_chunk) {
      Header *h = GetHeader(nearest_chunk, &header);
      CHECK_NE(h, nullptr);
      CHECK_GE(nearest_chunk, h->map_beg);
      CHECK_LT(nearest_chunk, h->map_beg + h->map_size);
      CHECK_LE(nearest_chunk, p);
      if (h->map_beg + h->map_size <= p)
        return nullptr;
    }
    return GetUser(nearest_chunk);
  }

  void EnsureSortedChunks() {
    if (chunks_sorted_)
      return;
    Sort(reinterpret_cast<uptr *>(chunks_), n_chunks_);
    chunks_sorted_ = true;
  }

  // This function does the same as GetBlockBegin, but is much faster.
  // Must be called with the allocator locked.
  void *GetBlockBeginFastLocked(const void *ptr) {
    if (!mem_funcs_inited_) return nullptr;
    mutex_.CheckLocked();
    uptr p = reinterpret_cast<uptr>(ptr);
    uptr n = n_chunks_;
    if (!n) return nullptr;
    EnsureSortedChunks();
    Header header, *h;
    h = GetHeader(chunks_[n - 1], &header);
    CHECK_NE(h, nullptr);
    uptr min_mmap_ = chunks_[0];
    uptr max_mmap_ = chunks_[n - 1] + h->map_size;
    if (p < min_mmap_ || p >= max_mmap_)
      return nullptr;
    uptr beg = 0, end = n - 1;
    // This loop is a log(n) lower_bound. It does not check for the exact match
    // to avoid expensive cache-thrashing loads.
    while (end - beg >= 2) {
      uptr mid = (beg + end) / 2;  // Invariant: mid >= beg + 1
      if (p < chunks_[mid])
        end = mid - 1;  // We are not interested in chunks[mid].
      else
        beg = mid;  // chunks[mid] may still be what we want.
    }

    if (beg < end) {
      CHECK_EQ(beg + 1, end);
      // There are 2 chunks left, choose one.
      if (p >= chunks_[end])
        beg = end;
    }

    if (p != chunks_[beg]) {
      h = GetHeader(chunks_[beg], &header);
      CHECK_NE(h, nullptr);
      if (h->map_beg + h->map_size <= p || p < h->map_beg)
        return nullptr;
    }
    return GetUser(chunks_[beg]);
  }

  void PrintStats() {
    Printf("Stats: DeviceAllocator: allocated %zd times, "
           "remains %zd (%zd K) max %zd M; by size logs: ",
           stats.n_allocs, stats.n_allocs - stats.n_frees,
           stats.currently_allocated >> 10, stats.max_allocated >> 20);
    for (uptr i = 0; i < ARRAY_SIZE(stats.by_size_log); i++) {
      uptr c = stats.by_size_log[i];
      if (!c) continue;
      Printf("%zd:%zd; ", i, c);
    }
    Printf("\n");
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() SANITIZER_ACQUIRE(mutex_) { mutex_.Lock(); }

  void ForceUnlock() SANITIZER_RELEASE(mutex_) { mutex_.Unlock(); }

  // Iterate over all existing chunks.
  // The allocator must be locked when calling this function.
  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    EnsureSortedChunks();  // Avoid doing the sort while iterating.
    for (uptr i = 0; i < n_chunks_; i++) {
      const uptr t = chunks_[i];
      callback(t, arg);
      // Consistency check: verify that the array did not change.
      CHECK_EQ(chunks_[i], t);
    }
  }

 private:
  bool InitMemFuncs() {
    if (!enabled_ || mem_funcs_inited_ || mem_funcs_init_count_ >= 2) {
      return mem_funcs_inited_;
    }
    mem_funcs_inited_ = DeviceMemFuncs::Init();
    mem_funcs_init_count_++;
    if (mem_funcs_inited_)
      page_size_ = DeviceMemFuncs::GetPageSize();
    return mem_funcs_inited_;
  }

  typedef DevivePointerInfo Header;

  Header *GetHeader(uptr p, Header* h) const {
    CHECK(IsAligned(p, page_size_));
    return DeviceMemFuncs::GetPointerInfo(p, h) ? h : nullptr;
  }

  void *GetUser(const uptr ptr) const {
    return reinterpret_cast<void *>(ptr);
  }

  uptr RoundUpMapSize(uptr size) {
    return RoundUpTo(size, page_size_) + page_size_;
  }

  bool enabled_;
  bool mem_funcs_inited_;
  // Maximum of mem_funcs_init_count_ is 2:
  //   1. The initial init called from Init(...), it could fail if
  //      libhsa-runtime64.so is dynamically loaded with dlopen()
  //   2. A potential deferred init called by Allocate(...)
  u32 mem_funcs_init_count_;
  uptr kMetadataSize_;
  uptr page_size_;
  uptr *chunks_;
  PtrArrayT ptr_array_;
  uptr n_chunks_;
  bool chunks_sorted_;
  struct Stats {
    uptr n_allocs, n_frees, currently_allocated, max_allocated, by_size_log[64];
  } stats;
  mutable StaticSpinMutex mutex_;
};
#endif  // SANITIZER_AMDGPU
