//===-- sanitizer_allocator_combined_device.h -------------------*- C++ -*-===//
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

// DeviceCombinedAllocator adds an optional device-memory heap tier on top of an
// existing host allocator, without touching the base class shared by every
// sanitizer. It is parameterized purely on the host allocator type (typically a
// CombinedAllocator of primary + secondary, but any allocator exposing the same
// interface works) and a DeviceAllocatorT (device backend, e.g. AMDGPU/HSA). It
// re-exposes the host allocator's public surface, dispatching to the device
// tier when a DeviceAllocationInfo is supplied (allocate) or when the pointer
// belongs to the device heap (everything else), and delegating to the host
// allocator otherwise. The host and device address ranges are disjoint, so "try
// host, then device" preserves the host allocator's dispatch semantics.
//
// Note: this composes (does not inherit from / modify) the host allocator, so
// it is independent of CombinedAllocator's definition. It still contains a host
// allocator because the front end (e.g. ASan) services host and device
// allocations through a single object that shares chunk/redzone metadata.
template <class HostAllocator, class DeviceAllocatorT>
class DeviceCombinedAllocator {
 public:
  using AllocatorCache = typename HostAllocator::AllocatorCache;
  using SecondaryAllocator = typename HostAllocator::SecondaryAllocator;

  void InitLinkerInitialized(s32 release_to_os_interval_ms,
                             uptr heap_start = 0) {
    device_stats_.Init();
    base_.InitLinkerInitialized(release_to_os_interval_ms, heap_start);
    device_.Init();
  }

  void Init(s32 release_to_os_interval_ms, uptr heap_start = 0) {
    device_stats_.Init();
    base_.Init(release_to_os_interval_ms, heap_start);
    device_.Init();
  }

  void* Allocate(AllocatorCache* cache, uptr size, uptr alignment,
                 DeviceAllocationInfo* da_info = nullptr) {
    if (!da_info)
      return base_.Allocate(cache, size, alignment);
    // Device tier: mirror CombinedAllocator's malloc(0)/overflow guards. The
    // device allocator handles its own alignment/page rounding, so 'size' is
    // forwarded unrounded (CombinedAllocator's original_size semantics).
    if (size == 0)
      size = 1;
    if (size + alignment < size) {
      Report(
          "WARNING: %s: DeviceCombinedAllocator allocation overflow: "
          "0x%zx bytes with 0x%zx alignment requested\n",
          SanitizerToolName, size, alignment);
      return nullptr;
    }
    void* res = device_.Allocate(&device_stats_, size, alignment, da_info);
    if (alignment > 8 && res)
      CHECK_EQ(reinterpret_cast<uptr>(res) & (alignment - 1), 0);
    return res;
  }

  s32 ReleaseToOSIntervalMs() const { return base_.ReleaseToOSIntervalMs(); }

  void SetReleaseToOSIntervalMs(s32 release_to_os_interval_ms) {
    base_.SetReleaseToOSIntervalMs(release_to_os_interval_ms);
  }

  void ForceReleaseToOS() { base_.ForceReleaseToOS(); }

  void Deallocate(AllocatorCache* cache, void* p) {
    if (!p)
      return;
    if (device_.PointerIsMine(p))
      device_.Deallocate(&device_stats_, p);
    else
      base_.Deallocate(cache, p);
  }

  void* Reallocate(AllocatorCache* cache, void* p, uptr new_size,
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
    void* new_p = Allocate(cache, new_size, alignment);
    if (new_p)
      internal_memcpy(new_p, p, memcpy_size);
    Deallocate(cache, p);
    return new_p;
  }

  bool PointerIsMine(const void* p) const {
    return base_.PointerIsMine(p) || device_.PointerIsMine(p);
  }

  bool FromPrimary(const void* p) const { return base_.FromPrimary(p); }

  void* GetMetaData(const void* p) {
    if (device_.PointerIsMine(p))
      return device_.GetMetaData(p);
    return base_.GetMetaData(p);
  }

  // Same as GetMetaData, but must be called with the allocator locked
  // (via ForceLock). Uses GetBlockBeginFastLocked for the device check to avoid
  // re-acquiring a mutex already held by the caller.
  void* GetMetaDataFastLocked(const void* p) {
    if (device_.GetBlockBeginFastLocked(p))
      return device_.GetMetaData(p);
    return base_.GetMetaDataFastLocked(p);
  }

  void* GetBlockBegin(const void* p) {
    if (void* beg = base_.GetBlockBegin(p))
      return beg;
    return device_.GetBlockBegin(p);
  }

  // This function does the same as GetBlockBegin, but is much faster.
  // Must be called with the allocator locked.
  void* GetBlockBeginFastLocked(const void* p) {
    if (void* beg = base_.GetBlockBeginFastLocked(p))
      return beg;
    return device_.GetBlockBeginFastLocked(p);
  }

  uptr GetActuallyAllocatedSize(void* p) {
    if (device_.PointerIsMine(p))
      return device_.GetActuallyAllocatedSize(p);
    return base_.GetActuallyAllocatedSize(p);
  }

  uptr TotalMemoryUsed() {
    return base_.TotalMemoryUsed() + device_.TotalMemoryUsed();
  }

  void TestOnlyUnmap() { base_.TestOnlyUnmap(); }

  void InitCache(AllocatorCache* cache) { base_.InitCache(cache); }

  void DestroyCache(AllocatorCache* cache) { base_.DestroyCache(cache); }

  void SwallowCache(AllocatorCache* cache) { base_.SwallowCache(cache); }

  void GetStats(AllocatorStatCounters s) const {
    base_.GetStats(s);
    AllocatorStatCounters device_counters;
    device_stats_.Get(device_counters);
    for (int i = 0; i < AllocatorStatCount; i++) s[i] += device_counters[i];
  }

  void PrintStats() {
    base_.PrintStats();
    device_.PrintStats();
  }

  // ForceLock() and ForceUnlock() are needed to implement Darwin malloc zone
  // introspection API.
  void ForceLock() SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
    device_.ForceLock();
    base_.ForceLock();
  }

  void ForceUnlock() SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
    base_.ForceUnlock();
    device_.ForceUnlock();
  }

  // Iterate over all existing chunks.
  // The allocator must be locked when calling this function.
  void ForEachChunk(ForEachChunkCallback callback, void* arg) {
    base_.ForEachChunk(callback, arg);
    device_.ForEachChunk(callback, arg);
  }

 private:
  HostAllocator base_;
  DeviceAllocatorT device_;
  AllocatorGlobalStats device_stats_;
};
