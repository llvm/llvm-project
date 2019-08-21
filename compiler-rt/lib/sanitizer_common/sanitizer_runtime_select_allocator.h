//===-- sanitizer_runtime_select_allocator.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Select one of the two allocators at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_RUNTIME_SELECT_ALLOCATOR_H
#define SANITIZER_RUNTIME_SELECT_ALLOCATOR_H

template <class Allocator1, class Allocator2>
class RuntimeSelectAllocator {
  Allocator1 a1;
  Allocator2 a2;

 public:
  bool use_first_allocator;

  class RuntimeSelectAllocatorCache {
    typename Allocator1::AllocatorCache a1;
    typename Allocator2::AllocatorCache a2;

   public:
    void Init(AllocatorGlobalStats *s) {
      if (this->use_first_allocator)
        a1.Init(s);
      else
        a2.Init(s);
    }
    void *Allocate(RuntimeSelectAllocator *allocator, uptr class_id) {
      if (allocator->use_first_allocator)
        return a1.Allocate(&allocator->a1, class_id);
      return a2.Allocate(&allocator->a2, class_id);
    }

    void Deallocate(RuntimeSelectAllocator *allocator, uptr class_id, void *p) {
      if (allocator->use_first_allocator)
        a1.Deallocate(&allocator->a1, class_id, p);
      else
        a2.Deallocate(&allocator->a2, class_id, p);
    }

    void Drain(RuntimeSelectAllocator *allocator) {
      if (allocator->use_first_allocator)
        a1.Drain(&allocator->a1);
      else
        a2.Drain(&allocator->a2);
    }

    void Destroy(RuntimeSelectAllocator *allocator, AllocatorGlobalStats *s) {
      if (allocator->use_first_allocator)
        a1.Destroy(&allocator->a1, s);
      else
        a2.Destroy(&allocator->a2, s);
    }
  };

  using MapUnmapCallback = typename Allocator1::MapUnmapCallback;
  using AddressSpaceView = typename Allocator1::AddressSpaceView;
  using AllocatorCache = RuntimeSelectAllocatorCache;

  void Init(s32 release_to_os_interval_ms) {
    // Use the first allocator when the address
    // space is too small for the 64-bit allocator.
    use_first_allocator = GetMaxVirtualAddress() < (((uptr)1ULL << 48) - 1);
    if (use_first_allocator)
      a1.Init(release_to_os_interval_ms);
    else
      a2.Init(release_to_os_interval_ms);
  }

  bool CanAllocate(uptr size, uptr alignment) {
    if (use_first_allocator)
      return Allocator1::CanAllocate(size, alignment);
    return Allocator2::CanAllocate(size, alignment);
  }

  uptr ClassID(uptr size) {
    if (use_first_allocator)
      return Allocator1::ClassID(size);
    return Allocator2::ClassID(size);
  }

  uptr KNumClasses() {
    if (use_first_allocator)
      return Allocator1::KNumClasses();
    return Allocator2::KNumClasses();
  }

  uptr KMaxSize() {
    if (use_first_allocator)
      return Allocator1::KMaxSize();
    return Allocator2::KMaxSize();
  }

  bool PointerIsMine(const void *p) {
    if (use_first_allocator)
      return a1.PointerIsMine(p);
    return a2.PointerIsMine(p);
  }

  void *GetMetaData(const void *p) {
    if (use_first_allocator)
      return a1.GetMetaData(p);
    return a2.GetMetaData(p);
  }

  uptr GetSizeClass(const void *p) {
    if (use_first_allocator)
      return a1.GetSizeClass(p);
    return a2.GetSizeClass(p);
  }

  void ForEachChunk(ForEachChunkCallback callback, void *arg) {
    if (use_first_allocator)
      a1.ForEachChunk(callback, arg);
    else
      a2.ForEachChunk(callback, arg);
  }

  void TestOnlyUnmap() {
    if (use_first_allocator)
      a1.TestOnlyUnmap();
    else
      a2.TestOnlyUnmap();
  }
  void ForceLock() {
    if (use_first_allocator)
      a1.ForceLock();
    else
      a2.ForceLock();
  }
  void ForceUnlock() {
    if (use_first_allocator)
      a1.ForceUnlock();
    else
      a2.ForceUnlock();
  }
  void *GetBlockBegin(const void *p) {
    if (use_first_allocator)
      return a1.GetBlockBegin(p);
    return a2.GetBlockBegin(p);
  }
  uptr GetActuallyAllocatedSize(void *p) {
    if (use_first_allocator)
      return a1.GetActuallyAllocatedSize(p);
    return a2.GetActuallyAllocatedSize(p);
  }
  void SetReleaseToOSIntervalMs(s32 release_to_os_interval_ms) {
    if (use_first_allocator)
      a1.SetReleaseToOSIntervalMs(release_to_os_interval_ms);
    else
      a2.SetReleaseToOSIntervalMs(release_to_os_interval_ms);
  }
  s32 ReleaseToOSIntervalMs() const {
    if (use_first_allocator)
      return a1.ReleaseToOSIntervalMs();
    return a2.ReleaseToOSIntervalMs();
  }
  void ForceReleaseToOS() {
    if (use_first_allocator)
      a1.ForceReleaseToOS();
    else
      a2.ForceReleaseToOS();
  }
  void PrintStats() {
    if (use_first_allocator)
      a1.PrintStats();
    else
      a2.PrintStats();
  }
};

#endif // SANITIZER_RUNTIME_SELECT_ALLOCATOR_H
