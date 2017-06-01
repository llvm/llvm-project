//===-- sanitizer_allocator_combined.h --------------------------*- C++ -*-===//
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

// This class implements a complete memory allocator by using two
// internal allocators:
// PrimaryAllocator is efficient, but may not allocate some sizes (alignments).
//  When allocating 2^x bytes it should return 2^x aligned chunk.
// PrimaryAllocator is used via a local AllocatorCache.
// SecondaryAllocator can allocate anything, but is not efficient.
template <class PrimaryAllocator, class AllocatorCache,
          class SecondaryAllocator>  // NOLINT
class CombinedAllocator {
 public:
  void InitCommon(bool may_return_null, s32 release_to_os_interval_ms) {
    primary_.Init(release_to_os_interval_ms);
    atomic_store(&may_return_null_, may_return_null, memory_order_relaxed);
  }

  void InitLinkerInitialized(
      bool may_return_null, s32 release_to_os_interval_ms) {
    secondary_.InitLinkerInitialized(may_return_null);
    stats_.InitLinkerInitialized();
    InitCommon(may_return_null, release_to_os_interval_ms);
  }

  void Init(bool may_return_null, s32 release_to_os_interval_ms) {
    secondary_.Init(may_return_null);
    stats_.Init();
    InitCommon(may_return_null, release_to_os_interval_ms);
  }

  void *Allocate(AllocatorCache *cache, uptr size, uptr alignment,
                 bool cleared = false, bool check_rss_limit = false) {
    // Returning 0 on malloc(0) may break a lot of code.
    if (size == 0)
      size = 1;
    if (size + alignment < size) return ReturnNullOrDieOnBadRequest();
    if (check_rss_limit && RssLimitIsExceeded()) return ReturnNullOrDieOnOOM();
    uptr original_size = size;
    // If alignment requirements are to be fulfilled by the frontend allocator
    // rather than by the primary or secondary, passing an alignment lower than
    // or equal to 8 will prevent any further rounding up, as well as the later
    // alignment check.
    if (alignment > 8)
      size = RoundUpTo(size, alignment);
    void *res;
    bool from_primary = primary_.CanAllocate(size, alignment);
    // The primary allocator should return a 2^x aligned allocation when
    // requested 2^x bytes, hence using the rounded up 'size' when being
    // serviced by the primary (this is no longer true when the primary is
    // using a non-fixed base address). The secondary takes care of the
    // alignment without such requirement, and allocating 'size' would use
    // extraneous memory, so we employ 'original_size'.
    if (from_primary)
      res = cache->Allocate(&primary_, primary_.ClassID(size));
    else
      res = secondary_.Allocate(&stats_, original_size, alignment);
    if (alignment > 8)
      CHECK_EQ(reinterpret_cast<uptr>(res) & (alignment - 1), 0);
    // When serviced by the secondary, the chunk comes from a mmap allocation
    // and will be zero'd out anyway. We only need to clear our the chunk if
    // it was serviced by the primary, hence using the rounded up 'size'.
    if (cleared && res && from_primary)
      internal_bzero_aligned16(res, RoundUpTo(size, 16));
    return res;
  }

  bool MayReturnNull() const {
    return atomic_load(&may_return_null_, memory_order_acquire);
  }

  void *ReturnNullOrDieOnBadRequest() {
    if (MayReturnNull())
      return nullptr;
    ReportAllocatorCannotReturnNull(false);
  }

  void *ReturnNullOrDieOnOOM() {
    if (MayReturnNull()) return nullptr;
    ReportAllocatorCannotReturnNull(true);
  }

  void SetMayReturnNull(bool may_return_null) {
    secondary_.SetMayReturnNull(may_return_null);
    atomic_store(&may_return_null_, may_return_null, memory_order_release);
  }

  s32 ReleaseToOSIntervalMs() const {
    return primary_.ReleaseToOSIntervalMs();
  }

  void SetReleaseToOSIntervalMs(s32 release_to_os_interval_ms) {
    primary_.SetReleaseToOSIntervalMs(release_to_os_interval_ms);
  }

  bool RssLimitIsExceeded() {
    return atomic_load(&rss_limit_is_exceeded_, memory_order_acquire);
  }

  void SetRssLimitIsExceeded(bool rss_limit_is_exceeded) {
    atomic_store(&rss_limit_is_exceeded_, rss_limit_is_exceeded,
                 memory_order_release);
  }

  void Deallocate(AllocatorCache *cache, void *p) {
    if (!p) return;
    if (primary_.PointerIsMine(p))
      cache->Deallocate(&primary_, primary_.GetSizeClass(p), p);
    else
      secondary_.Deallocate(&stats_, p);
  }

  void *Reallocate(AllocatorCache *cache, void *p, uptr new_size,
                   uptr alignment) {
    if (!p)
      return Allocate(cache, new_size, alignment);
    if (!new_size) {
      Deallocate(cache, p);
      return nullptr;
    }
    CHECK(PointerIsMine(p));
    uptr old_size = GetActuallyAllocatedSize(p);
    uptr memcpy_size = Min(new_size, old_size);
    void *new_p = Allocate(cache, new_size, alignment);
    if (new_p)
      internal_memcpy(new_p, p, memcpy_size);
    Deallocate(cache, p);
    return new_p;
  }

  bool PointerIsMine(void *p) {
    if (primary_.PointerIsMine(p))
      return true;
    return secondary_.PointerIsMine(p);
  }

  bool FromPrimary(void *p) {
    return primary_.PointerIsMine(p);
  }

  void *GetMetaData(const void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetMetaData(p);
    return secondary_.GetMetaData(p);
  }

  void *GetBlockBegin(const void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetBlockBegin(p);
    return secondary_.GetBlockBegin(p);
  }

  // This function does the same as GetBlockBegin, but is much faster.
  // Must be called with the allocator locked.
  void *GetBlockBeginFastLocked(void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetBlockBegin(p);
    return secondary_.GetBlockBeginFastLocked(p);
  }

  uptr GetActuallyAllocatedSize(void *p) {
    if (primary_.PointerIsMine(p))
      return primary_.GetActuallyAllocatedSize(p);
    return secondary_.GetActuallyAllocatedSize(p);
  }

  uptr TotalMemoryUsed() {
    return primary_.TotalMemoryUsed() + secondary_.TotalMemoryUsed();
  }

  void TestOnlyUnmap() { primary_.TestOnlyUnmap(); }

  void InitCache(AllocatorCache *cache) {
    cache->Init(&stats_);
  }

  void DestroyCache(AllocatorCache *cache) {
    cache->Destroy(&primary_, &stats_);
  }

  void SwallowCache(AllocatorCache *cache) {
    cache->Drain(&primary_);
  }

  void GetStats(AllocatorStatCounters s) const {
    stats_.Get(s);
  }

  void PrintStats() {
    primary_.PrintStats();
    secondary_.PrintStats();
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() {
    primary_.ForceLock();
    secondary_.ForceLock();
  }

  void ForceUnlock() {
    secondary_.ForceUnlock();
    primary_.ForceUnlock();
  }

  // Iterate over all existing chunks.
  // The allocator must be locked when calling this function.
  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    primary_.ForEachChunk(callback, arg);
    secondary_.ForEachChunk(callback, arg);
  }

 private:
  PrimaryAllocator primary_;
  SecondaryAllocator secondary_;
  AllocatorGlobalStats stats_;
  atomic_uint8_t may_return_null_;
  atomic_uint8_t rss_limit_is_exceeded_;
};

