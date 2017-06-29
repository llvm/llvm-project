//===-- scudo_allocator_combined.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo Combined Allocator, dispatches allocation & deallocation requests to
/// the Primary or the Secondary backend allocators.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ALLOCATOR_COMBINED_H_
#define SCUDO_ALLOCATOR_COMBINED_H_

#ifndef SCUDO_ALLOCATOR_H_
#error "This file must be included inside scudo_allocator.h."
#endif

template <class PrimaryAllocator, class AllocatorCache,
    class SecondaryAllocator>
class ScudoCombinedAllocator {
 public:
  void Init(s32 ReleaseToOSIntervalMs) {
    Primary.Init(ReleaseToOSIntervalMs);
    Secondary.Init();
    Stats.Init();
  }

  void *Allocate(AllocatorCache *Cache, uptr Size, uptr Alignment,
                 bool FromPrimary) {
    if (FromPrimary)
      return Cache->Allocate(&Primary, Primary.ClassID(Size));
    return Secondary.Allocate(&Stats, Size, Alignment);
  }

  void Deallocate(AllocatorCache *Cache, void *Ptr, bool FromPrimary) {
    if (FromPrimary)
      Cache->Deallocate(&Primary, Primary.GetSizeClass(Ptr), Ptr);
    else
      Secondary.Deallocate(&Stats, Ptr);
  }

  uptr GetActuallyAllocatedSize(void *Ptr, bool FromPrimary) {
    if (FromPrimary)
      return PrimaryAllocator::ClassIdToSize(Primary.GetSizeClass(Ptr));
    return Secondary.GetActuallyAllocatedSize(Ptr);
  }

  void InitCache(AllocatorCache *Cache) {
    Cache->Init(&Stats);
  }

  void DestroyCache(AllocatorCache *Cache) {
    Cache->Destroy(&Primary, &Stats);
  }

  void GetStats(AllocatorStatCounters StatType) const {
    Stats.Get(StatType);
  }

 private:
  PrimaryAllocator Primary;
  SecondaryAllocator Secondary;
  AllocatorGlobalStats Stats;
};

#endif  // SCUDO_ALLOCATOR_COMBINED_H_
