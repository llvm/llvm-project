//===-- sanitizer_allocator_primary32.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#error This file must be included inside sanitizer_allocator.h
#endif

// SizeClassAllocator32 -- allocator for 32-bit address space.
// This allocator can theoretically be used on 64-bit arch, but there it is less
// efficient than SizeClassAllocator64.
//
// [kSpaceBeg, kSpaceBeg + kSpaceSize) is the range of addresses which can
// be returned by MmapOrDie().
//
// Region:
//   a result of a single call to MmapAlignedOrDie(kRegionSize, kRegionSize).
// Since the regions are aligned by kRegionSize, there are exactly
// kNumPossibleRegions possible regions in the address space and so we keep
// a ByteMap possible_regions to store the size classes of each Region.
// 0 size class means the region is not used by the allocator.
//
// One Region is used to allocate chunks of a single size class.
// A Region looks like this:
// UserChunk1 .. UserChunkN <gap> MetaChunkN .. MetaChunk1
//
// In order to avoid false sharing the objects of this class should be
// chache-line aligned.
template <const uptr kSpaceBeg, const u64 kSpaceSize,
          const uptr kMetadataSize, class SizeClassMap,
          const uptr kRegionSizeLog,
          class ByteMap,
          class MapUnmapCallback = NoOpMapUnmapCallback>
class SizeClassAllocator32 {
 public:
  typedef typename SizeClassMap::TransferBatch Batch;
  typedef SizeClassAllocator32<kSpaceBeg, kSpaceSize, kMetadataSize,
      SizeClassMap, kRegionSizeLog, ByteMap, MapUnmapCallback> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> AllocatorCache;

  void Init() {
    possible_regions.TestOnlyInit();
    internal_memset(size_class_info_array, 0, sizeof(size_class_info_array));
  }

  void *MapWithCallback(uptr size) {
    size = RoundUpTo(size, GetPageSizeCached());
    void *res = MmapOrDie(size, "SizeClassAllocator32");
    MapUnmapCallback().OnMap((uptr)res, size);
    return res;
  }

  void UnmapWithCallback(uptr beg, uptr size) {
    MapUnmapCallback().OnUnmap(beg, size);
    UnmapOrDie(reinterpret_cast<void *>(beg), size);
  }

  static bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  void *GetMetaData(const void *p) {
    CHECK(PointerIsMine(p));
    uptr mem = reinterpret_cast<uptr>(p);
    uptr beg = ComputeRegionBeg(mem);
    uptr size = SizeClassMap::Size(GetSizeClass(p));
    u32 offset = mem - beg;
    uptr n = offset / (u32)size;  // 32-bit division
    uptr meta = (beg + kRegionSize) - (n + 1) * kMetadataSize;
    return reinterpret_cast<void*>(meta);
  }

  NOINLINE Batch* AllocateBatch(AllocatorStats *stat, AllocatorCache *c,
                                uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    SizeClassInfo *sci = GetSizeClassInfo(class_id);
    SpinMutexLock l(&sci->mutex);
    if (sci->free_list.empty())
      PopulateFreeList(stat, c, sci, class_id);
    CHECK(!sci->free_list.empty());
    Batch *b = sci->free_list.front();
    sci->free_list.pop_front();
    return b;
  }

  NOINLINE void DeallocateBatch(AllocatorStats *stat, uptr class_id, Batch *b) {
    CHECK_LT(class_id, kNumClasses);
    SizeClassInfo *sci = GetSizeClassInfo(class_id);
    SpinMutexLock l(&sci->mutex);
    CHECK_GT(b->count, 0);
    sci->free_list.push_front(b);
  }

  bool PointerIsMine(const void *p) {
    uptr mem = reinterpret_cast<uptr>(p);
    if (mem < kSpaceBeg || mem >= kSpaceBeg + kSpaceSize)
      return false;
    return GetSizeClass(p) != 0;
  }

  uptr GetSizeClass(const void *p) {
    return possible_regions[ComputeRegionId(reinterpret_cast<uptr>(p))];
  }

  void *GetBlockBegin(const void *p) {
    CHECK(PointerIsMine(p));
    uptr mem = reinterpret_cast<uptr>(p);
    uptr beg = ComputeRegionBeg(mem);
    uptr size = SizeClassMap::Size(GetSizeClass(p));
    u32 offset = mem - beg;
    u32 n = offset / (u32)size;  // 32-bit division
    uptr res = beg + (n * (u32)size);
    return reinterpret_cast<void*>(res);
  }

  uptr GetActuallyAllocatedSize(void *p) {
    CHECK(PointerIsMine(p));
    return SizeClassMap::Size(GetSizeClass(p));
  }

  uptr ClassID(uptr size) { return SizeClassMap::ClassID(size); }

  uptr TotalMemoryUsed() {
    // No need to lock here.
    uptr res = 0;
    for (uptr i = 0; i < kNumPossibleRegions; i++)
      if (possible_regions[i])
        res += kRegionSize;
    return res;
  }

  void TestOnlyUnmap() {
    for (uptr i = 0; i < kNumPossibleRegions; i++)
      if (possible_regions[i])
        UnmapWithCallback((i * kRegionSize), kRegionSize);
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    for (uptr i = 0; i < kNumClasses; i++) {
      GetSizeClassInfo(i)->mutex.Lock();
    }
  }

  void ForceUnlock() {
    for (int i = kNumClasses - 1; i >= 0; i--) {
      GetSizeClassInfo(i)->mutex.Unlock();
    }
  }

  // Iterate over all existing chunks.
  // The allocator must be locked when calling this function.
  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    for (uptr region = 0; region < kNumPossibleRegions; region++)
      if (possible_regions[region]) {
        uptr chunk_size = SizeClassMap::Size(possible_regions[region]);
        uptr max_chunks_in_region = kRegionSize / (chunk_size + kMetadataSize);
        uptr region_beg = region * kRegionSize;
        for (uptr chunk = region_beg;
             chunk < region_beg + max_chunks_in_region * chunk_size;
             chunk += chunk_size) {
          // Too slow: CHECK_EQ((void *)chunk, GetBlockBegin((void *)chunk));
          callback(chunk, arg);
        }
      }
  }

  void PrintStats() {
  }

  static uptr AdditionalSize() {
    return 0;
  }

  typedef SizeClassMap SizeClassMapT;
  static const uptr kNumClasses = SizeClassMap::kNumClasses;

 private:
  static const uptr kRegionSize = 1 << kRegionSizeLog;
  static const uptr kNumPossibleRegions = kSpaceSize / kRegionSize;

  struct SizeClassInfo {
    SpinMutex mutex;
    IntrusiveList<Batch> free_list;
    char padding[kCacheLineSize - sizeof(uptr) - sizeof(IntrusiveList<Batch>)];
  };
  COMPILER_CHECK(sizeof(SizeClassInfo) == kCacheLineSize);

  uptr ComputeRegionId(uptr mem) {
    uptr res = mem >> kRegionSizeLog;
    CHECK_LT(res, kNumPossibleRegions);
    return res;
  }

  uptr ComputeRegionBeg(uptr mem) {
    return mem & ~(kRegionSize - 1);
  }

  uptr AllocateRegion(AllocatorStats *stat, uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    uptr res = reinterpret_cast<uptr>(MmapAlignedOrDie(kRegionSize, kRegionSize,
                                      "SizeClassAllocator32"));
    MapUnmapCallback().OnMap(res, kRegionSize);
    stat->Add(AllocatorStatMapped, kRegionSize);
    CHECK_EQ(0U, (res & (kRegionSize - 1)));
    possible_regions.set(ComputeRegionId(res), static_cast<u8>(class_id));
    return res;
  }

  SizeClassInfo *GetSizeClassInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    return &size_class_info_array[class_id];
  }

  void PopulateFreeList(AllocatorStats *stat, AllocatorCache *c,
                        SizeClassInfo *sci, uptr class_id) {
    uptr size = SizeClassMap::Size(class_id);
    uptr reg = AllocateRegion(stat, class_id);
    uptr n_chunks = kRegionSize / (size + kMetadataSize);
    uptr max_count = SizeClassMap::MaxCached(class_id);
    Batch *b = nullptr;
    for (uptr i = reg; i < reg + n_chunks * size; i += size) {
      if (!b) {
        if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
          b = (Batch*)c->Allocate(this, SizeClassMap::ClassID(sizeof(Batch)));
        else
          b = (Batch*)i;
        b->count = 0;
      }
      b->batch[b->count++] = (void*)i;
      if (b->count == max_count) {
        CHECK_GT(b->count, 0);
        sci->free_list.push_back(b);
        b = nullptr;
      }
    }
    if (b) {
      CHECK_GT(b->count, 0);
      sci->free_list.push_back(b);
    }
  }

  ByteMap possible_regions;
  SizeClassInfo size_class_info_array[kNumClasses];
};


