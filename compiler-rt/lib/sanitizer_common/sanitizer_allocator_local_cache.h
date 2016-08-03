//===-- sanitizer_allocator_local_cache.h -----------------------*- C++ -*-===//
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

// Objects of this type should be used as local caches for SizeClassAllocator64
// or SizeClassAllocator32. Since the typical use of this class is to have one
// object per thread in TLS, is has to be POD.
template<class SizeClassAllocator>
struct SizeClassAllocatorLocalCache {
  typedef SizeClassAllocator Allocator;
  static const uptr kNumClasses = SizeClassAllocator::kNumClasses;

  void Init(AllocatorGlobalStats *s) {
    stats_.Init();
    if (s)
      s->Register(&stats_);
  }

  void Destroy(SizeClassAllocator *allocator, AllocatorGlobalStats *s) {
    Drain(allocator);
    if (s)
      s->Unregister(&stats_);
  }

  void *Allocate(SizeClassAllocator *allocator, uptr class_id) {
    CHECK_NE(class_id, 0UL);
    CHECK_LT(class_id, kNumClasses);
    stats_.Add(AllocatorStatAllocated, SizeClassMap::Size(class_id));
    PerClass *c = &per_class_[class_id];
    if (UNLIKELY(c->count == 0))
      Refill(allocator, class_id);
    void *res = c->batch[--c->count];
    PREFETCH(c->batch[c->count - 1]);
    return res;
  }

  void Deallocate(SizeClassAllocator *allocator, uptr class_id, void *p) {
    CHECK_NE(class_id, 0UL);
    CHECK_LT(class_id, kNumClasses);
    // If the first allocator call on a new thread is a deallocation, then
    // max_count will be zero, leading to check failure.
    InitCache();
    stats_.Sub(AllocatorStatAllocated, SizeClassMap::Size(class_id));
    PerClass *c = &per_class_[class_id];
    CHECK_NE(c->max_count, 0UL);
    if (UNLIKELY(c->count == c->max_count))
      Drain(allocator, class_id);
    c->batch[c->count++] = p;
  }

  void Drain(SizeClassAllocator *allocator) {
    for (uptr class_id = 0; class_id < kNumClasses; class_id++) {
      PerClass *c = &per_class_[class_id];
      while (c->count > 0)
        Drain(allocator, class_id);
    }
  }

  // private:
  typedef typename SizeClassAllocator::SizeClassMapT SizeClassMap;
  typedef typename SizeClassMap::TransferBatch Batch;
  struct PerClass {
    uptr count;
    uptr max_count;
    void *batch[2 * SizeClassMap::kMaxNumCached];
  };
  PerClass per_class_[kNumClasses];
  AllocatorStats stats_;

  void InitCache() {
    if (per_class_[1].max_count)
      return;
    for (uptr i = 0; i < kNumClasses; i++) {
      PerClass *c = &per_class_[i];
      c->max_count = 2 * SizeClassMap::MaxCached(i);
    }
  }

  // Returns a Batch suitable for class_id.
  // For small size classes allocates the batch from the allocator.
  // For large size classes simply returns b.
  Batch *CreateBatch(uptr class_id, SizeClassAllocator *allocator, Batch *b) {
    if (uptr batch_class_id = SizeClassMap::SizeClassForTransferBatch(class_id))
      return (Batch*)Allocate(allocator, batch_class_id);
    return b;
  }

  // Destroys Batch b.
  // For small size classes deallocates b to the allocator.
  // Does notthing for large size classes.
  void DestroyBatch(uptr class_id, SizeClassAllocator *allocator, Batch *b) {
    if (uptr batch_class_id = SizeClassMap::SizeClassForTransferBatch(class_id))
      Deallocate(allocator, batch_class_id, b);
  }

  NOINLINE void Refill(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    Batch *b = allocator->AllocateBatch(&stats_, this, class_id);
    CHECK_GT(b->Count(), 0);
    for (uptr i = 0; i < b->Count(); i++)
      c->batch[i] = b->Get(i);
    c->count = b->Count();
    DestroyBatch(class_id, allocator, b);
  }

  NOINLINE void Drain(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    uptr cnt = Min(c->max_count / 2, c->count);
    uptr first_idx_to_drain = c->count - cnt;
    Batch *b =
        CreateBatch(class_id, allocator, (Batch *)c->batch[first_idx_to_drain]);
    b->SetFromArray(&c->batch[first_idx_to_drain], cnt);
    c->count -= cnt;
    allocator->DeallocateBatch(&stats_, class_id, b);
  }
};
