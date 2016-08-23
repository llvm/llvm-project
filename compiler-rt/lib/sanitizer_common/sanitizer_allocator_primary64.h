//===-- sanitizer_allocator_primary64.h -------------------------*- C++ -*-===//
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

template<class SizeClassAllocator> struct SizeClassAllocator64LocalCache;

// SizeClassAllocator64 -- allocator for 64-bit address space.
//
// Space: a portion of address space of kSpaceSize bytes starting at SpaceBeg.
// If kSpaceBeg is ~0 then SpaceBeg is chosen dynamically my mmap.
// Otherwise SpaceBeg=kSpaceBeg (fixed address).
// kSpaceSize is a power of two.
// At the beginning the entire space is mprotect-ed, then small parts of it
// are mapped on demand.
//
// Region: a part of Space dedicated to a single size class.
// There are kNumClasses Regions of equal size.
//
// UserChunk: a piece of memory returned to user.
// MetaChunk: kMetadataSize bytes of metadata associated with a UserChunk.
//
// A Region looks like this:
// UserChunk1 ... UserChunkN <gap> MetaChunkN ... MetaChunk1
template <const uptr kSpaceBeg, const uptr kSpaceSize,
          const uptr kMetadataSize, class SizeClassMap,
          class MapUnmapCallback = NoOpMapUnmapCallback>
class SizeClassAllocator64 {
 public:
  struct TransferBatch {
    static const uptr kMaxNumCached = SizeClassMap::kMaxNumCachedHint - 4;
    void SetFromRange(uptr region_beg, uptr beg_offset, uptr step, uptr count) {
      count_ = count;
      CHECK_LE(count_, kMaxNumCached);
      region_beg_ = region_beg;
      for (uptr i = 0; i < count; i++)
        batch_[i] = static_cast<u32>((beg_offset + i * step) >> 4);
    }
    void SetFromArray(uptr region_beg, void *batch[], uptr count) {
      count_ = count;
      CHECK_LE(count_, kMaxNumCached);
      region_beg_ = region_beg;
      for (uptr i = 0; i < count; i++)
        batch_[i] = static_cast<u32>(
            ((reinterpret_cast<uptr>(batch[i])) - region_beg) >> 4);
    }
    void CopyToArray(void *to_batch[]) {
      for (uptr i = 0, n = Count(); i < n; i++)
        to_batch[i] = reinterpret_cast<void*>(Get(i));
    }
    uptr Count() const { return count_; }

    // How much memory do we need for a batch containing n elements.
    static uptr AllocationSizeRequiredForNElements(uptr n) {
      return sizeof(uptr) * 2 + sizeof(u32) * n;
    }
    static uptr MaxCached(uptr class_id) {
      return Min(kMaxNumCached, SizeClassMap::MaxCachedHint(class_id));
    }

    TransferBatch *next;

   private:
    uptr Get(uptr i) {
      return region_beg_ + (static_cast<uptr>(batch_[i]) << 4);
    }
    // Instead of storing 64-bit pointers we store 32-bit offsets from the
    // region start divided by 4. This imposes two limitations:
    // * all allocations are 16-aligned,
    // * regions are not larger than 2^36.
    uptr region_beg_ : SANITIZER_WORDSIZE - 10;  // Region-beg is 4096-aligned.
    uptr count_      : 10;
    u32 batch_[kMaxNumCached];
  };
  static const uptr kBatchSize = sizeof(TransferBatch);
  COMPILER_CHECK((kBatchSize & (kBatchSize - 1)) == 0);
  COMPILER_CHECK(sizeof(TransferBatch) ==
                 SizeClassMap::kMaxNumCachedHint * sizeof(u32));
  COMPILER_CHECK(TransferBatch::kMaxNumCached < 1024);  // count_ uses 10 bits.

  static uptr ClassIdToSize(uptr class_id) {
    return class_id == SizeClassMap::kBatchClassID
               ? sizeof(TransferBatch)
               : SizeClassMap::Size(class_id);
  }

  typedef SizeClassAllocator64<kSpaceBeg, kSpaceSize, kMetadataSize,
      SizeClassMap, MapUnmapCallback> ThisT;
  typedef SizeClassAllocator64LocalCache<ThisT> AllocatorCache;

  void Init() {
    uptr TotalSpaceSize = kSpaceSize + AdditionalSize();
    if (kUsingConstantSpaceBeg) {
      CHECK_EQ(kSpaceBeg, reinterpret_cast<uptr>(
                              MmapFixedNoAccess(kSpaceBeg, TotalSpaceSize)));
    } else {
      NonConstSpaceBeg =
          reinterpret_cast<uptr>(MmapNoAccess(TotalSpaceSize));
      CHECK_NE(NonConstSpaceBeg, ~(uptr)0);
    }
    MapWithCallback(SpaceEnd(), AdditionalSize());
  }

  void MapWithCallback(uptr beg, uptr size) {
    CHECK_EQ(beg, reinterpret_cast<uptr>(MmapFixedOrDie(beg, size)));
    MapUnmapCallback().OnMap(beg, size);
  }

  void UnmapWithCallback(uptr beg, uptr size) {
    MapUnmapCallback().OnUnmap(beg, size);
    UnmapOrDie(reinterpret_cast<void *>(beg), size);
  }

  static bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  NOINLINE TransferBatch *AllocateBatch(AllocatorStats *stat, AllocatorCache *c,
                                        uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    TransferBatch *b = region->free_list.Pop();
    if (!b)
      b = PopulateFreeList(stat, c, class_id, region);
    region->n_allocated += b->Count();
    return b;
  }

  NOINLINE void DeallocateBatch(AllocatorStats *stat, uptr class_id,
                                TransferBatch *b) {
    RegionInfo *region = GetRegionInfo(class_id);
    CHECK_GT(b->Count(), 0);
    region->free_list.Push(b);
    region->n_freed += b->Count();
  }

  bool PointerIsMine(const void *p) {
    uptr P = reinterpret_cast<uptr>(p);
    if (kUsingConstantSpaceBeg && (kSpaceBeg % kSpaceSize) == 0)
      return P / kSpaceSize == kSpaceBeg / kSpaceSize;
    return P >= SpaceBeg() && P < SpaceEnd();
  }

  uptr GetRegionBegin(const void *p) {
    if (kUsingConstantSpaceBeg)
      return reinterpret_cast<uptr>(p) & ~(kRegionSize - 1);
    uptr space_beg = SpaceBeg();
    return ((reinterpret_cast<uptr>(p)  - space_beg) & ~(kRegionSize - 1)) +
        space_beg;
  }

  uptr GetRegionBeginBySizeClass(uptr class_id) {
    return SpaceBeg() + kRegionSize * class_id;
  }

  uptr GetSizeClass(const void *p) {
    if (kUsingConstantSpaceBeg && (kSpaceBeg % kSpaceSize) == 0)
      return ((reinterpret_cast<uptr>(p)) / kRegionSize) % kNumClassesRounded;
    return ((reinterpret_cast<uptr>(p) - SpaceBeg()) / kRegionSize) %
           kNumClassesRounded;
  }

  void *GetBlockBegin(const void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = ClassIdToSize(class_id);
    if (!size) return nullptr;
    uptr chunk_idx = GetChunkIdx((uptr)p, size);
    uptr reg_beg = GetRegionBegin(p);
    uptr beg = chunk_idx * size;
    uptr next_beg = beg + size;
    if (class_id >= kNumClasses) return nullptr;
    RegionInfo *region = GetRegionInfo(class_id);
    if (region->mapped_user >= next_beg)
      return reinterpret_cast<void*>(reg_beg + beg);
    return nullptr;
  }

  uptr GetActuallyAllocatedSize(void *p) {
    CHECK(PointerIsMine(p));
    return ClassIdToSize(GetSizeClass(p));
  }

  uptr ClassID(uptr size) { return SizeClassMap::ClassID(size); }

  void *GetMetaData(const void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = ClassIdToSize(class_id);
    uptr chunk_idx = GetChunkIdx(reinterpret_cast<uptr>(p), size);
    return reinterpret_cast<void *>(SpaceBeg() +
                                    (kRegionSize * (class_id + 1)) -
                                    (1 + chunk_idx) * kMetadataSize);
  }

  uptr TotalMemoryUsed() {
    uptr res = 0;
    for (uptr i = 0; i < kNumClasses; i++)
      res += GetRegionInfo(i)->allocated_user;
    return res;
  }

  // Test-only.
  void TestOnlyUnmap() {
    UnmapWithCallback(SpaceBeg(), kSpaceSize + AdditionalSize());
  }

  static void FillMemoryProfile(uptr start, uptr rss, bool file, uptr *stats,
                           uptr stats_size) {
    for (uptr class_id = 0; class_id < stats_size; class_id++)
      if (stats[class_id] == start)
        stats[class_id] = rss;
  }

  void PrintStats() {
    uptr total_mapped = 0;
    uptr n_allocated = 0;
    uptr n_freed = 0;
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      total_mapped += region->mapped_user;
      n_allocated += region->n_allocated;
      n_freed += region->n_freed;
    }
    Printf("Stats: SizeClassAllocator64: %zdM mapped in %zd allocations; "
           "remains %zd\n",
           total_mapped >> 20, n_allocated, n_allocated - n_freed);
    uptr rss_stats[kNumClasses];
    for (uptr class_id = 0; class_id < kNumClasses; class_id++)
      rss_stats[class_id] = SpaceBeg() + kRegionSize * class_id;
    GetMemoryProfile(FillMemoryProfile, rss_stats, kNumClasses);
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      if (region->mapped_user == 0) continue;
      uptr in_use = region->n_allocated - region->n_freed;
      uptr avail_chunks = region->allocated_user / ClassIdToSize(class_id);
      Printf("  %02zd (%zd): mapped: %zdK allocs: %zd frees: %zd inuse: %zd"
             " avail: %zd rss: %zdK\n",
             class_id,
             ClassIdToSize(class_id),
             region->mapped_user >> 10,
             region->n_allocated,
             region->n_freed,
             in_use, avail_chunks,
             rss_stats[class_id] >> 10);
    }
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    for (uptr i = 0; i < kNumClasses; i++) {
      GetRegionInfo(i)->mutex.Lock();
    }
  }

  void ForceUnlock() {
    for (int i = (int)kNumClasses - 1; i >= 0; i--) {
      GetRegionInfo(i)->mutex.Unlock();
    }
  }

  // Iterate over all existing chunks.
  // The allocator must be locked when calling this function.
  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      uptr chunk_size = ClassIdToSize(class_id);
      uptr region_beg = SpaceBeg() + class_id * kRegionSize;
      for (uptr chunk = region_beg;
           chunk < region_beg + region->allocated_user;
           chunk += chunk_size) {
        // Too slow: CHECK_EQ((void *)chunk, GetBlockBegin((void *)chunk));
        callback(chunk, arg);
      }
    }
  }

  static uptr AdditionalSize() {
    return RoundUpTo(sizeof(RegionInfo) * kNumClassesRounded,
                     GetPageSizeCached());
  }

  typedef SizeClassMap SizeClassMapT;
  static const uptr kNumClasses = SizeClassMap::kNumClasses;
  static const uptr kNumClassesRounded = SizeClassMap::kNumClassesRounded;

 private:
  static const uptr kRegionSize = kSpaceSize / kNumClassesRounded;

  static const bool kUsingConstantSpaceBeg = kSpaceBeg != ~(uptr)0;
  uptr NonConstSpaceBeg;
  uptr SpaceBeg() const {
    return kUsingConstantSpaceBeg ? kSpaceBeg : NonConstSpaceBeg;
  }
  uptr SpaceEnd() const { return  SpaceBeg() + kSpaceSize; }
  // kRegionSize must be >= 2^32.
  COMPILER_CHECK((kRegionSize) >= (1ULL << (SANITIZER_WORDSIZE / 2)));
  // kRegionSize must be <= 2^36, see TransferBatch.
  COMPILER_CHECK((kRegionSize) <= (1ULL << (SANITIZER_WORDSIZE / 2 + 4)));
  // Call mmap for user memory with at least this size.
  static const uptr kUserMapSize = 1 << 16;
  // Call mmap for metadata memory with at least this size.
  static const uptr kMetaMapSize = 1 << 16;

  struct RegionInfo {
    BlockingMutex mutex;
    LFStack<TransferBatch> free_list;
    uptr allocated_user;  // Bytes allocated for user memory.
    uptr allocated_meta;  // Bytes allocated for metadata.
    uptr mapped_user;  // Bytes mapped for user memory.
    uptr mapped_meta;  // Bytes mapped for metadata.
    uptr n_allocated, n_freed;  // Just stats.
  };
  COMPILER_CHECK(sizeof(RegionInfo) >= kCacheLineSize);

  RegionInfo *GetRegionInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *regions =
        reinterpret_cast<RegionInfo *>(SpaceBeg() + kSpaceSize);
    return &regions[class_id];
  }

  uptr GetChunkIdx(uptr chunk, uptr size) {
    if (!kUsingConstantSpaceBeg)
      chunk -= SpaceBeg();

    uptr offset = chunk % kRegionSize;
    // Here we divide by a non-constant. This is costly.
    // size always fits into 32-bits. If the offset fits too, use 32-bit div.
    if (offset >> (SANITIZER_WORDSIZE / 2))
      return offset / size;
    return (u32)offset / (u32)size;
  }

  NOINLINE TransferBatch *PopulateFreeList(AllocatorStats *stat,
                                           AllocatorCache *c, uptr class_id,
                                           RegionInfo *region) {
    BlockingMutexLock l(&region->mutex);
    TransferBatch *b = region->free_list.Pop();
    if (b)
      return b;
    uptr size = ClassIdToSize(class_id);
    uptr count = TransferBatch::MaxCached(class_id);
    uptr beg_idx = region->allocated_user;
    uptr end_idx = beg_idx + count * size;
    uptr region_beg = SpaceBeg() + kRegionSize * class_id;
    if (end_idx + size > region->mapped_user) {
      // Do the mmap for the user memory.
      uptr map_size = kUserMapSize;
      while (end_idx + size > region->mapped_user + map_size)
        map_size += kUserMapSize;
      CHECK_GE(region->mapped_user + map_size, end_idx);
      MapWithCallback(region_beg + region->mapped_user, map_size);
      stat->Add(AllocatorStatMapped, map_size);
      region->mapped_user += map_size;
    }
    uptr total_count = (region->mapped_user - beg_idx - size)
        / size / count * count;
    region->allocated_meta += total_count * kMetadataSize;
    if (region->allocated_meta > region->mapped_meta) {
      uptr map_size = kMetaMapSize;
      while (region->allocated_meta > region->mapped_meta + map_size)
        map_size += kMetaMapSize;
      // Do the mmap for the metadata.
      CHECK_GE(region->mapped_meta + map_size, region->allocated_meta);
      MapWithCallback(region_beg + kRegionSize -
                      region->mapped_meta - map_size, map_size);
      region->mapped_meta += map_size;
    }
    CHECK_LE(region->allocated_meta, region->mapped_meta);
    if (region->mapped_user + region->mapped_meta > kRegionSize) {
      Printf("%s: Out of memory. Dying. ", SanitizerToolName);
      Printf("The process has exhausted %zuMB for size class %zu.\n",
          kRegionSize / 1024 / 1024, size);
      Die();
    }
    for (;;) {
      b = c->CreateBatch(class_id, this,
                         (TransferBatch *)(region_beg + beg_idx));
      b->SetFromRange(region_beg, beg_idx, size, count);
      region->allocated_user += count * size;
      CHECK_LE(region->allocated_user, region->mapped_user);
      beg_idx += count * size;
      if (beg_idx + count * size + size > region->mapped_user)
        break;
      CHECK_GT(b->Count(), 0);
      region->free_list.Push(b);
    }
    return b;
  }
};


