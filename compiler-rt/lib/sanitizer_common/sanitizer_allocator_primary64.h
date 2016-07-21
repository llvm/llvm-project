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
  typedef typename SizeClassMap::TransferBatch Batch;
  typedef SizeClassAllocator64<kSpaceBeg, kSpaceSize, kMetadataSize,
      SizeClassMap, MapUnmapCallback> ThisT;
  typedef SizeClassAllocatorLocalCache<ThisT> AllocatorCache;

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

  NOINLINE Batch* AllocateBatch(AllocatorStats *stat, AllocatorCache *c,
                                uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *region = GetRegionInfo(class_id);
    Batch *b = region->free_list.Pop();
    if (!b)
      b = PopulateFreeList(stat, c, class_id, region);
    region->n_allocated += b->count;
    return b;
  }

  NOINLINE void DeallocateBatch(AllocatorStats *stat, uptr class_id, Batch *b) {
    RegionInfo *region = GetRegionInfo(class_id);
    CHECK_GT(b->count, 0);
    region->free_list.Push(b);
    region->n_freed += b->count;
  }

  bool PointerIsMine(const void *p) {
    uptr P = reinterpret_cast<uptr>(p);
    if (kUsingConstantSpaceBeg && (kSpaceBeg % kSpaceSize) == 0)
      return P / kSpaceSize == kSpaceBeg / kSpaceSize;
    return P >= SpaceBeg() && P < SpaceEnd();
  }

  uptr GetSizeClass(const void *p) {
    if (kUsingConstantSpaceBeg && (kSpaceBeg % kSpaceSize) == 0)
      return ((reinterpret_cast<uptr>(p)) / kRegionSize) % kNumClassesRounded;
    return ((reinterpret_cast<uptr>(p) - SpaceBeg()) / kRegionSize) %
           kNumClassesRounded;
  }

  void *GetBlockBegin(const void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = SizeClassMap::Size(class_id);
    if (!size) return nullptr;
    uptr chunk_idx = GetChunkIdx((uptr)p, size);
    uptr reg_beg = (uptr)p & ~(kRegionSize - 1);
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
    return SizeClassMap::Size(GetSizeClass(p));
  }

  uptr ClassID(uptr size) { return SizeClassMap::ClassID(size); }

  void *GetMetaData(const void *p) {
    uptr class_id = GetSizeClass(p);
    uptr size = SizeClassMap::Size(class_id);
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
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      if (region->mapped_user == 0) continue;
      Printf("  %02zd (%zd): total: %zd K allocs: %zd remains: %zd\n",
             class_id,
             SizeClassMap::Size(class_id),
             region->mapped_user >> 10,
             region->n_allocated,
             region->n_allocated - region->n_freed);
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
      uptr chunk_size = SizeClassMap::Size(class_id);
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
  // Populate the free list with at most this number of bytes at once
  // or with one element if its size is greater.
  static const uptr kPopulateSize = 1 << 14;
  // Call mmap for user memory with at least this size.
  static const uptr kUserMapSize = 1 << 16;
  // Call mmap for metadata memory with at least this size.
  static const uptr kMetaMapSize = 1 << 16;

  struct RegionInfo {
    BlockingMutex mutex;
    LFStack<Batch> free_list;
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

  static uptr GetChunkIdx(uptr chunk, uptr size) {
    uptr offset = chunk % kRegionSize;
    // Here we divide by a non-constant. This is costly.
    // size always fits into 32-bits. If the offset fits too, use 32-bit div.
    if (offset >> (SANITIZER_WORDSIZE / 2))
      return offset / size;
    return (u32)offset / (u32)size;
  }

  NOINLINE Batch* PopulateFreeList(AllocatorStats *stat, AllocatorCache *c,
                                   uptr class_id, RegionInfo *region) {
    BlockingMutexLock l(&region->mutex);
    Batch *b = region->free_list.Pop();
    if (b)
      return b;
    uptr size = SizeClassMap::Size(class_id);
    uptr count = size < kPopulateSize ? SizeClassMap::MaxCached(class_id) : 1;
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
      if (SizeClassMap::SizeClassRequiresSeparateTransferBatch(class_id))
        b = (Batch*)c->Allocate(this, SizeClassMap::ClassID(sizeof(Batch)));
      else
        b = (Batch*)(region_beg + beg_idx);
      b->count = count;
      for (uptr i = 0; i < count; i++)
        b->batch[i] = (void*)(region_beg + beg_idx + i * size);
      region->allocated_user += count * size;
      CHECK_LE(region->allocated_user, region->mapped_user);
      beg_idx += count * size;
      if (beg_idx + count * size + size > region->mapped_user)
        break;
      CHECK_GT(b->count, 0);
      region->free_list.Push(b);
    }
    return b;
  }
};


