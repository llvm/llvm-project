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
// The template parameter Params is a class containing the actual parameters.
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

// FreeArray is an array free-d chunks (stored as 4-byte offsets)
//
// A Region looks like this:
// UserChunk1 ... UserChunkN <gap> MetaChunkN ... MetaChunk1 FreeArray

struct SizeClassAllocator64FlagMasks {  //  Bit masks.
  enum {
    kRandomShuffleChunks = 1,
  };
};

template <class Params>
class SizeClassAllocator64 {
 public:
  static const uptr kSpaceBeg = Params::kSpaceBeg;
  static const uptr kSpaceSize = Params::kSpaceSize;
  static const uptr kMetadataSize = Params::kMetadataSize;
  typedef typename Params::SizeClassMap SizeClassMap;
  typedef typename Params::MapUnmapCallback MapUnmapCallback;

  static const bool kRandomShuffleChunks =
      Params::kFlags & SizeClassAllocator64FlagMasks::kRandomShuffleChunks;

  typedef SizeClassAllocator64<Params> ThisT;
  typedef SizeClassAllocator64LocalCache<ThisT> AllocatorCache;

  // When we know the size class (the region base) we can represent a pointer
  // as a 4-byte integer (offset from the region start shifted right by 4).
  typedef u32 CompactPtrT;
  static const uptr kCompactPtrScale = 4;
  CompactPtrT PointerToCompactPtr(uptr base, uptr ptr) {
    return static_cast<CompactPtrT>((ptr - base) >> kCompactPtrScale);
  }
  uptr CompactPtrToPointer(uptr base, CompactPtrT ptr32) {
    return base + (static_cast<uptr>(ptr32) << kCompactPtrScale);
  }

  void Init(s32 release_to_os_interval_ms) {
    uptr TotalSpaceSize = kSpaceSize + AdditionalSize();
    if (kUsingConstantSpaceBeg) {
      CHECK_EQ(kSpaceBeg, reinterpret_cast<uptr>(
                              MmapFixedNoAccess(kSpaceBeg, TotalSpaceSize)));
    } else {
      NonConstSpaceBeg =
          reinterpret_cast<uptr>(MmapNoAccess(TotalSpaceSize));
      CHECK_NE(NonConstSpaceBeg, ~(uptr)0);
    }
    SetReleaseToOSIntervalMs(release_to_os_interval_ms);
    MapWithCallbackOrDie(SpaceEnd(), AdditionalSize());
  }

  s32 ReleaseToOSIntervalMs() const {
    return atomic_load(&release_to_os_interval_ms_, memory_order_relaxed);
  }

  void SetReleaseToOSIntervalMs(s32 release_to_os_interval_ms) {
    atomic_store(&release_to_os_interval_ms_, release_to_os_interval_ms,
                 memory_order_relaxed);
  }

  static bool CanAllocate(uptr size, uptr alignment) {
    return size <= SizeClassMap::kMaxSize &&
      alignment <= SizeClassMap::kMaxSize;
  }

  NOINLINE void ReturnToAllocator(AllocatorStats *stat, uptr class_id,
                                  const CompactPtrT *chunks, uptr n_chunks) {
    RegionInfo *region = GetRegionInfo(class_id);
    uptr region_beg = GetRegionBeginBySizeClass(class_id);
    CompactPtrT *free_array = GetFreeArray(region_beg);

    BlockingMutexLock l(&region->mutex);
    uptr old_num_chunks = region->num_freed_chunks;
    uptr new_num_freed_chunks = old_num_chunks + n_chunks;
    // Failure to allocate free array space while releasing memory is non
    // recoverable.
    if (UNLIKELY(!EnsureFreeArraySpace(region, region_beg,
                                       new_num_freed_chunks)))
      DieOnFailure::OnOOM();
    for (uptr i = 0; i < n_chunks; i++)
      free_array[old_num_chunks + i] = chunks[i];
    region->num_freed_chunks = new_num_freed_chunks;
    region->stats.n_freed += n_chunks;

    MaybeReleaseToOS(class_id);
  }

  NOINLINE bool GetFromAllocator(AllocatorStats *stat, uptr class_id,
                                 CompactPtrT *chunks, uptr n_chunks) {
    RegionInfo *region = GetRegionInfo(class_id);
    uptr region_beg = GetRegionBeginBySizeClass(class_id);
    CompactPtrT *free_array = GetFreeArray(region_beg);

    BlockingMutexLock l(&region->mutex);
    if (UNLIKELY(region->num_freed_chunks < n_chunks)) {
      if (UNLIKELY(!PopulateFreeArray(stat, class_id, region,
                                      n_chunks - region->num_freed_chunks)))
        return false;
      CHECK_GE(region->num_freed_chunks, n_chunks);
    }
    region->num_freed_chunks -= n_chunks;
    uptr base_idx = region->num_freed_chunks;
    for (uptr i = 0; i < n_chunks; i++)
      chunks[i] = free_array[base_idx + i];
    region->stats.n_allocated += n_chunks;
    return true;
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
    uptr region_beg = GetRegionBeginBySizeClass(class_id);
    return reinterpret_cast<void *>(GetMetadataEnd(region_beg) -
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
    UnmapWithCallbackOrDie(SpaceBeg(), kSpaceSize + AdditionalSize());
  }

  static void FillMemoryProfile(uptr start, uptr rss, bool file, uptr *stats,
                           uptr stats_size) {
    for (uptr class_id = 0; class_id < stats_size; class_id++)
      if (stats[class_id] == start)
        stats[class_id] = rss;
  }

  void PrintStats(uptr class_id, uptr rss) {
    RegionInfo *region = GetRegionInfo(class_id);
    if (region->mapped_user == 0) return;
    uptr in_use = region->stats.n_allocated - region->stats.n_freed;
    uptr avail_chunks = region->allocated_user / ClassIdToSize(class_id);
    Printf(
        "%s %02zd (%6zd): mapped: %6zdK allocs: %7zd frees: %7zd inuse: %6zd "
        "num_freed_chunks %7zd avail: %6zd rss: %6zdK releases: %6zd\n",
        region->exhausted ? "F" : " ", class_id, ClassIdToSize(class_id),
        region->mapped_user >> 10, region->stats.n_allocated,
        region->stats.n_freed, in_use, region->num_freed_chunks, avail_chunks,
        rss >> 10, region->rtoi.num_releases);
  }

  void PrintStats() {
    uptr total_mapped = 0;
    uptr n_allocated = 0;
    uptr n_freed = 0;
    for (uptr class_id = 1; class_id < kNumClasses; class_id++) {
      RegionInfo *region = GetRegionInfo(class_id);
      total_mapped += region->mapped_user;
      n_allocated += region->stats.n_allocated;
      n_freed += region->stats.n_freed;
    }
    Printf("Stats: SizeClassAllocator64: %zdM mapped in %zd allocations; "
           "remains %zd\n",
           total_mapped >> 20, n_allocated, n_allocated - n_freed);
    uptr rss_stats[kNumClasses];
    for (uptr class_id = 0; class_id < kNumClasses; class_id++)
      rss_stats[class_id] = SpaceBeg() + kRegionSize * class_id;
    GetMemoryProfile(FillMemoryProfile, rss_stats, kNumClasses);
    for (uptr class_id = 1; class_id < kNumClasses; class_id++)
      PrintStats(class_id, rss_stats[class_id]);
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

  static uptr ClassIdToSize(uptr class_id) {
    return SizeClassMap::Size(class_id);
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
  // FreeArray is the array of free-d chunks (stored as 4-byte offsets).
  // In the worst case it may reguire kRegionSize/SizeClassMap::kMinSize
  // elements, but in reality this will not happen. For simplicity we
  // dedicate 1/8 of the region's virtual space to FreeArray.
  static const uptr kFreeArraySize = kRegionSize / 8;

  static const bool kUsingConstantSpaceBeg = kSpaceBeg != ~(uptr)0;
  uptr NonConstSpaceBeg;
  uptr SpaceBeg() const {
    return kUsingConstantSpaceBeg ? kSpaceBeg : NonConstSpaceBeg;
  }
  uptr SpaceEnd() const { return  SpaceBeg() + kSpaceSize; }
  // kRegionSize must be >= 2^32.
  COMPILER_CHECK((kRegionSize) >= (1ULL << (SANITIZER_WORDSIZE / 2)));
  // kRegionSize must be <= 2^36, see CompactPtrT.
  COMPILER_CHECK((kRegionSize) <= (1ULL << (SANITIZER_WORDSIZE / 2 + 4)));
  // Call mmap for user memory with at least this size.
  static const uptr kUserMapSize = 1 << 16;
  // Call mmap for metadata memory with at least this size.
  static const uptr kMetaMapSize = 1 << 16;
  // Call mmap for free array memory with at least this size.
  static const uptr kFreeArrayMapSize = 1 << 16;

  atomic_sint32_t release_to_os_interval_ms_;

  struct Stats {
    uptr n_allocated;
    uptr n_freed;
  };

  struct ReleaseToOsInfo {
    uptr n_freed_at_last_release;
    uptr num_releases;
    u64 last_release_at_ns;
  };

  struct RegionInfo {
    BlockingMutex mutex;
    uptr num_freed_chunks;  // Number of elements in the freearray.
    uptr mapped_free_array;  // Bytes mapped for freearray.
    uptr allocated_user;  // Bytes allocated for user memory.
    uptr allocated_meta;  // Bytes allocated for metadata.
    uptr mapped_user;  // Bytes mapped for user memory.
    uptr mapped_meta;  // Bytes mapped for metadata.
    u32 rand_state;  // Seed for random shuffle, used if kRandomShuffleChunks.
    bool exhausted;  // Whether region is out of space for new chunks.
    Stats stats;
    ReleaseToOsInfo rtoi;
  };
  COMPILER_CHECK(sizeof(RegionInfo) >= kCacheLineSize);

  u32 Rand(u32 *state) {  // ANSI C linear congruential PRNG.
    return (*state = *state * 1103515245 + 12345) >> 16;
  }

  u32 RandN(u32 *state, u32 n) { return Rand(state) % n; }  // [0, n)

  void RandomShuffle(u32 *a, u32 n, u32 *rand_state) {
    if (n <= 1) return;
    for (u32 i = n - 1; i > 0; i--)
      Swap(a[i], a[RandN(rand_state, i + 1)]);
  }

  RegionInfo *GetRegionInfo(uptr class_id) {
    CHECK_LT(class_id, kNumClasses);
    RegionInfo *regions =
        reinterpret_cast<RegionInfo *>(SpaceBeg() + kSpaceSize);
    return &regions[class_id];
  }

  uptr GetMetadataEnd(uptr region_beg) {
    return region_beg + kRegionSize - kFreeArraySize;
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

  CompactPtrT *GetFreeArray(uptr region_beg) {
    return reinterpret_cast<CompactPtrT *>(region_beg + kRegionSize -
                                           kFreeArraySize);
  }

  bool MapWithCallback(uptr beg, uptr size) {
    uptr mapped = reinterpret_cast<uptr>(MmapFixedOrDieOnFatalError(beg, size));
    if (UNLIKELY(!mapped))
      return false;
    CHECK_EQ(beg, mapped);
    MapUnmapCallback().OnMap(beg, size);
    return true;
  }

  void MapWithCallbackOrDie(uptr beg, uptr size) {
    CHECK_EQ(beg, reinterpret_cast<uptr>(MmapFixedOrDie(beg, size)));
    MapUnmapCallback().OnMap(beg, size);
  }

  void UnmapWithCallbackOrDie(uptr beg, uptr size) {
    MapUnmapCallback().OnUnmap(beg, size);
    UnmapOrDie(reinterpret_cast<void *>(beg), size);
  }

  bool EnsureFreeArraySpace(RegionInfo *region, uptr region_beg,
                            uptr num_freed_chunks) {
    uptr needed_space = num_freed_chunks * sizeof(CompactPtrT);
    if (region->mapped_free_array < needed_space) {
      CHECK_LE(needed_space, kFreeArraySize);
      uptr new_mapped_free_array = RoundUpTo(needed_space, kFreeArrayMapSize);
      uptr current_map_end = reinterpret_cast<uptr>(GetFreeArray(region_beg)) +
                             region->mapped_free_array;
      uptr new_map_size = new_mapped_free_array - region->mapped_free_array;
      if (UNLIKELY(!MapWithCallback(current_map_end, new_map_size)))
        return false;
      region->mapped_free_array = new_mapped_free_array;
    }
    return true;
  }

  NOINLINE bool PopulateFreeArray(AllocatorStats *stat, uptr class_id,
                                  RegionInfo *region, uptr requested_count) {
    // region->mutex is held.
    const uptr size = ClassIdToSize(class_id);
    const uptr new_space_beg = region->allocated_user;
    const uptr new_space_end = new_space_beg + requested_count * size;
    const uptr region_beg = GetRegionBeginBySizeClass(class_id);

    // Map more space for chunks, if necessary.
    if (new_space_end > region->mapped_user) {
      if (!kUsingConstantSpaceBeg && region->mapped_user == 0)
        region->rand_state = static_cast<u32>(region_beg >> 12);  // From ASLR.
      // Do the mmap for the user memory.
      uptr map_size = kUserMapSize;
      while (new_space_end > region->mapped_user + map_size)
        map_size += kUserMapSize;
      CHECK_GE(region->mapped_user + map_size, new_space_end);
      if (UNLIKELY(!MapWithCallback(region_beg + region->mapped_user,
                                    map_size)))
        return false;
      stat->Add(AllocatorStatMapped, map_size);
      region->mapped_user += map_size;
    }
    const uptr new_chunks_count = (region->mapped_user - new_space_beg) / size;

    // Calculate the required space for metadata.
    const uptr requested_allocated_meta =
        region->allocated_meta + new_chunks_count * kMetadataSize;
    uptr requested_mapped_meta = region->mapped_meta;
    while (requested_allocated_meta > requested_mapped_meta)
      requested_mapped_meta += kMetaMapSize;
    // Check whether this size class is exhausted.
    if (region->mapped_user + requested_mapped_meta >
        kRegionSize - kFreeArraySize) {
      if (!region->exhausted) {
        region->exhausted = true;
        Printf("%s: Out of memory. ", SanitizerToolName);
        Printf("The process has exhausted %zuMB for size class %zu.\n",
               kRegionSize >> 20, size);
      }
      return false;
    }
    // Map more space for metadata, if necessary.
    if (requested_mapped_meta > region->mapped_meta) {
      if (UNLIKELY(!MapWithCallback(
              GetMetadataEnd(region_beg) - requested_mapped_meta,
              requested_mapped_meta - region->mapped_meta)))
        return false;
      region->mapped_meta = requested_mapped_meta;
    }

    // If necessary, allocate more space for the free array and populate it with
    // newly allocated chunks.
    const uptr total_freed_chunks = region->num_freed_chunks + new_chunks_count;
    if (UNLIKELY(!EnsureFreeArraySpace(region, region_beg, total_freed_chunks)))
      return false;
    CompactPtrT *free_array = GetFreeArray(region_beg);
    for (uptr i = 0, chunk = new_space_beg; i < new_chunks_count;
         i++, chunk += size)
      free_array[total_freed_chunks - 1 - i] = PointerToCompactPtr(0, chunk);
    if (kRandomShuffleChunks)
      RandomShuffle(&free_array[region->num_freed_chunks], new_chunks_count,
                    &region->rand_state);

    // All necessary memory is mapped and now it is safe to advance all
    // 'allocated_*' counters.
    region->num_freed_chunks += new_chunks_count;
    region->allocated_user += new_chunks_count * size;
    CHECK_LE(region->allocated_user, region->mapped_user);
    region->allocated_meta = requested_allocated_meta;
    CHECK_LE(region->allocated_meta, region->mapped_meta);
    region->exhausted = false;

    return true;
  }

  void MaybeReleaseChunkRange(uptr region_beg, uptr chunk_size,
                              CompactPtrT first, CompactPtrT last) {
    uptr beg_ptr = CompactPtrToPointer(region_beg, first);
    uptr end_ptr = CompactPtrToPointer(region_beg, last) + chunk_size;
    ReleaseMemoryPagesToOS(beg_ptr, end_ptr);
  }

  // Attempts to release some RAM back to OS. The region is expected to be
  // locked.
  // Algorithm:
  // * Sort the chunks.
  // * Find ranges fully covered by free-d chunks
  // * Release them to OS with madvise.
  void MaybeReleaseToOS(uptr class_id) {
    RegionInfo *region = GetRegionInfo(class_id);
    const uptr chunk_size = ClassIdToSize(class_id);
    const uptr page_size = GetPageSizeCached();

    uptr n = region->num_freed_chunks;
    if (n * chunk_size < page_size)
      return;  // No chance to release anything.
    if ((region->stats.n_freed -
         region->rtoi.n_freed_at_last_release) * chunk_size < page_size) {
      return;  // Nothing new to release.
    }

    s32 interval_ms = ReleaseToOSIntervalMs();
    if (interval_ms < 0)
      return;

    u64 now_ns = NanoTime();
    if (region->rtoi.last_release_at_ns + interval_ms * 1000000ULL > now_ns)
      return;  // Memory was returned recently.
    region->rtoi.last_release_at_ns = now_ns;

    uptr region_beg = GetRegionBeginBySizeClass(class_id);
    CompactPtrT *free_array = GetFreeArray(region_beg);
    SortArray(free_array, n);

    const uptr scaled_chunk_size = chunk_size >> kCompactPtrScale;
    const uptr kScaledGranularity = page_size >> kCompactPtrScale;

    uptr range_beg = free_array[0];
    uptr prev = free_array[0];
    for (uptr i = 1; i < n; i++) {
      uptr chunk = free_array[i];
      CHECK_GT(chunk, prev);
      if (chunk - prev != scaled_chunk_size) {
        CHECK_GT(chunk - prev, scaled_chunk_size);
        if (prev + scaled_chunk_size - range_beg >= kScaledGranularity) {
          MaybeReleaseChunkRange(region_beg, chunk_size, range_beg, prev);
          region->rtoi.n_freed_at_last_release = region->stats.n_freed;
          region->rtoi.num_releases++;
        }
        range_beg = chunk;
      }
      prev = chunk;
    }
  }
};
