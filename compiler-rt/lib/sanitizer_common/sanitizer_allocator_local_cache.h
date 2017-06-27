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
struct SizeClassAllocatorLocalCache
    : SizeClassAllocator::AllocatorCache {
};

// Cache used by SizeClassAllocator64.
template <class SizeClassAllocator>
struct SizeClassAllocator64LocalCache {
  typedef SizeClassAllocator Allocator;
  static const uptr kNumClasses = SizeClassAllocator::kNumClasses;
  typedef typename Allocator::SizeClassMapT SizeClassMap;
  typedef typename Allocator::CompactPtrT CompactPtrT;

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
    PerClass *c = &per_class_[class_id];
    if (UNLIKELY(c->count == 0)) {
      if (UNLIKELY(!Refill(c, allocator, class_id)))
        return nullptr;
    }
    stats_.Add(AllocatorStatAllocated, c->class_size);
    CHECK_GT(c->count, 0);
    CompactPtrT chunk = c->chunks[--c->count];
    void *res = reinterpret_cast<void *>(allocator->CompactPtrToPointer(
        allocator->GetRegionBeginBySizeClass(class_id), chunk));
    return res;
  }

  void Deallocate(SizeClassAllocator *allocator, uptr class_id, void *p) {
    CHECK_NE(class_id, 0UL);
    CHECK_LT(class_id, kNumClasses);
    // If the first allocator call on a new thread is a deallocation, then
    // max_count will be zero, leading to check failure.
    InitCache();
    PerClass *c = &per_class_[class_id];
    stats_.Sub(AllocatorStatAllocated, c->class_size);
    CHECK_NE(c->max_count, 0UL);
    if (UNLIKELY(c->count == c->max_count))
      Drain(c, allocator, class_id, c->max_count / 2);
    CompactPtrT chunk = allocator->PointerToCompactPtr(
        allocator->GetRegionBeginBySizeClass(class_id),
        reinterpret_cast<uptr>(p));
    c->chunks[c->count++] = chunk;
  }

  void Drain(SizeClassAllocator *allocator) {
    for (uptr class_id = 0; class_id < kNumClasses; class_id++) {
      PerClass *c = &per_class_[class_id];
      while (c->count > 0)
        Drain(c, allocator, class_id, c->count);
    }
  }

  // private:
  struct PerClass {
    u32 count;
    u32 max_count;
    uptr class_size;
    CompactPtrT chunks[2 * SizeClassMap::kMaxNumCachedHint];
  };
  PerClass per_class_[kNumClasses];
  AllocatorStats stats_;

  void InitCache() {
    if (per_class_[1].max_count)
      return;
    for (uptr i = 0; i < kNumClasses; i++) {
      PerClass *c = &per_class_[i];
      c->max_count = 2 * SizeClassMap::MaxCachedHint(i);
      c->class_size = Allocator::ClassIdToSize(i);
    }
  }

  NOINLINE bool Refill(PerClass *c, SizeClassAllocator *allocator,
                       uptr class_id) {
    InitCache();
    uptr num_requested_chunks = c->max_count / 2;
    if (UNLIKELY(!allocator->GetFromAllocator(&stats_, class_id, c->chunks,
                                              num_requested_chunks)))
      return false;
    c->count = num_requested_chunks;
    return true;
  }

  NOINLINE void Drain(PerClass *c, SizeClassAllocator *allocator, uptr class_id,
                      uptr count) {
    InitCache();
    CHECK_GE(c->count, count);
    uptr first_idx_to_drain = c->count - count;
    c->count -= count;
    allocator->ReturnToAllocator(&stats_, class_id,
                                 &c->chunks[first_idx_to_drain], count);
  }
};

// Cache used by SizeClassAllocator32.
template <class SizeClassAllocator>
struct SizeClassAllocator32LocalCache {
  typedef SizeClassAllocator Allocator;
  typedef typename Allocator::TransferBatch TransferBatch;
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
    PerClass *c = &per_class_[class_id];
    if (UNLIKELY(c->count == 0)) {
      if (UNLIKELY(!Refill(allocator, class_id)))
        return nullptr;
    }
    stats_.Add(AllocatorStatAllocated, c->class_size);
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
    PerClass *c = &per_class_[class_id];
    stats_.Sub(AllocatorStatAllocated, c->class_size);
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
  struct PerClass {
    uptr count;
    uptr max_count;
    uptr class_size;
    uptr class_id_for_transfer_batch;
    void *batch[2 * TransferBatch::kMaxNumCached];
  };
  PerClass per_class_[kNumClasses];
  AllocatorStats stats_;

  void InitCache() {
    if (per_class_[1].max_count)
      return;
    // TransferBatch class is declared in SizeClassAllocator.
    uptr class_id_for_transfer_batch =
        SizeClassMap::ClassID(sizeof(TransferBatch));
    for (uptr i = 0; i < kNumClasses; i++) {
      PerClass *c = &per_class_[i];
      uptr max_cached = TransferBatch::MaxCached(i);
      c->max_count = 2 * max_cached;
      c->class_size = Allocator::ClassIdToSize(i);
      // We transfer chunks between central and thread-local free lists in
      // batches. For small size classes we allocate batches separately. For
      // large size classes we may use one of the chunks to store the batch.
      // sizeof(TransferBatch) must be a power of 2 for more efficient
      // allocation.
      c->class_id_for_transfer_batch = (c->class_size <
          TransferBatch::AllocationSizeRequiredForNElements(max_cached)) ?
              class_id_for_transfer_batch : 0;
    }
  }

  // Returns a TransferBatch suitable for class_id.
  // For small size classes allocates the batch from the allocator.
  // For large size classes simply returns b.
  TransferBatch *CreateBatch(uptr class_id, SizeClassAllocator *allocator,
                             TransferBatch *b) {
    if (uptr batch_class_id = per_class_[class_id].class_id_for_transfer_batch)
      return (TransferBatch*)Allocate(allocator, batch_class_id);
    return b;
  }

  // Destroys TransferBatch b.
  // For small size classes deallocates b to the allocator.
  // Does notthing for large size classes.
  void DestroyBatch(uptr class_id, SizeClassAllocator *allocator,
                    TransferBatch *b) {
    if (uptr batch_class_id = per_class_[class_id].class_id_for_transfer_batch)
      Deallocate(allocator, batch_class_id, b);
  }

  NOINLINE bool Refill(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    TransferBatch *b = allocator->AllocateBatch(&stats_, this, class_id);
    if (UNLIKELY(!b))
      return false;
    CHECK_GT(b->Count(), 0);
    b->CopyToArray(c->batch);
    c->count = b->Count();
    DestroyBatch(class_id, allocator, b);
    return true;
  }

  NOINLINE void Drain(SizeClassAllocator *allocator, uptr class_id) {
    InitCache();
    PerClass *c = &per_class_[class_id];
    uptr cnt = Min(c->max_count / 2, c->count);
    uptr first_idx_to_drain = c->count - cnt;
    TransferBatch *b = CreateBatch(
        class_id, allocator, (TransferBatch *)c->batch[first_idx_to_drain]);
    // Failure to allocate a batch while releasing memory is non recoverable.
    // TODO(alekseys): Figure out how to do it without allocating a new batch.
    if (UNLIKELY(!b))
      DieOnFailure::OnOOM();
    b->SetFromArray(allocator->GetRegionBeginBySizeClass(class_id),
                    &c->batch[first_idx_to_drain], cnt);
    c->count -= cnt;
    allocator->DeallocateBatch(&stats_, class_id, b);
  }
};

