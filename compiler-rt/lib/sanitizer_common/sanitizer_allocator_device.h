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
    CheckAndInitMemFuncs(false);
  }

  void *Allocate(AllocatorStats *stat, uptr size, uptr alignment,
                 DeviceAllocationInfo *da_info) {
    if (!da_info || !CheckAndInitMemFuncs(false))
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
    uptr map_beg, map_size;
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
      chunks_[idx] = chunks_[--n_chunks_];
      chunks_sorted_ = false;
      stats.n_frees++;
      map_beg = p_;
      bool ret = DeviceMemFuncs::GetBlockBeginEnd(p, nullptr, &end);
      CHECK_EQ(ret, true);
      map_size = end - map_beg + 1;
      stats.currently_allocated -= map_size;
      stat->Sub(AllocatorStatAllocated, map_size);
      stat->Sub(AllocatorStatMapped, map_size);
    }
    MapUnmapCallback().OnUnmap(map_beg, map_size);
    DeviceMemFuncs::Deallocate(p);
  }

  bool PointerIsMine(const void *p) { return GetBlockBegin(p) != nullptr; }

  void *GetBlockBegin(const void *ptr) {
    if (!CheckAndInitMemFuncs())
      return nullptr;
    SpinMutexLock l(&mutex_);
    return GetBlockBeginFastLocked(const_cast<void *>(ptr));
  }

  void *GetBlockBeginFastLocked(void *ptr) {
    if (!CheckAndInitMemFuncs())
      return nullptr;

    uptr ptr_ = reinterpret_cast<uptr>(ptr);
    mutex_.CheckLocked();
    EnsureSortedChunks();  // Avoid doing the sort while iterating.
    uptr idx;
    for (idx = 0; idx < n_chunks_; idx++) {
      if (chunks_[idx] >= ptr_)
        break;
    }
    if (idx < n_chunks_ && chunks_[idx] == ptr_)
      return ptr;
    if (idx == 0)
      return nullptr;

    void *p = reinterpret_cast<void *>(chunks_[idx - 1]);
    uptr end;
    if (!DeviceMemFuncs::GetBlockBeginEnd(p, nullptr, &end) || ptr_ >= end)
      return nullptr;
    else
      return p;
  }

  void *GetMetaData(const void *p) {
    uptr end;
    if (!DeviceMemFuncs::GetBlockBeginEnd(p, nullptr, &end))
      return nullptr;
    else
      return reinterpret_cast<void *>(end - kMetadataSize_);
  }

  uptr GetActuallyAllocatedSize(void *p) {
    uptr beg, end;
    if (DeviceMemFuncs::GetBlockBeginEnd(p, &beg, &end))
      return end - beg;
    else
      return 0;
  }

  void ForceLock() { mutex_.Lock(); }
  void ForceUnlock() { mutex_.Unlock(); }

  void EnsureSortedChunks() {
    if (chunks_sorted_)
      return;
    Sort(reinterpret_cast<uptr *>(chunks_), n_chunks_);
    chunks_sorted_ = true;
  }

  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    EnsureSortedChunks();  // Avoid doing the sort while iterating.
    for (uptr i = 0; i < n_chunks_; i++) {
      const uptr t = chunks_[i];
      callback(t, arg);
      // Consistency check: verify that the array did not change.
      CHECK_EQ(chunks_[i], t);
    }
  }

  uptr TotalMemoryUsed() {
    SpinMutexLock l(&mutex_);
    uptr res = 0, beg, end;
    for (uptr i = 0; i < n_chunks_; i++) {
      void *p = chunks_[i];
      DeviceMemFuncs::GetBlockBeginEnd(p, &beg, &end);
      res += RoundUpMapSize(end - beg + 1);
    }
    return res;
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

 private:
  bool CheckAndInitMemFuncs(bool check_only = true) {
    if (!enabled_ ||
        check_only ||
        mem_funcs_inited_ ||
        mem_funcs_init_count_ >= 2) {
      return mem_funcs_inited_;
    }
    mem_funcs_inited_ = DeviceMemFuncs::Init();
    mem_funcs_init_count_++;
    if (mem_funcs_inited_)
      page_size_ = DeviceMemFuncs::GetPageSize();
    return mem_funcs_inited_;
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
  StaticSpinMutex mutex_;
};
#endif  // SANITIZER_AMDGPU
