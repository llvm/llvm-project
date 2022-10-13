//===-- local_cache.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LOCAL_CACHE_H_
#define SCUDO_LOCAL_CACHE_H_

#include "internal_defs.h"
#include "platform.h"
#include "report.h"
#include "stats.h"

namespace scudo {

template <class SizeClassAllocator> struct SizeClassAllocatorLocalCache {
  typedef typename SizeClassAllocator::SizeClassMap SizeClassMap;
  typedef typename SizeClassAllocator::CompactPtrT CompactPtrT;

  struct TransferBatch {
    static const u16 MaxNumCached = SizeClassMap::MaxNumCachedHint;
    void setFromArray(CompactPtrT *Array, u16 N) {
      DCHECK_LE(N, MaxNumCached);
      Count = N;
      memcpy(Batch, Array, sizeof(Batch[0]) * Count);
    }
    void clear() { Count = 0; }
    void add(CompactPtrT P) {
      DCHECK_LT(Count, MaxNumCached);
      Batch[Count++] = P;
    }
    void copyToArray(CompactPtrT *Array) const {
      memcpy(Array, Batch, sizeof(Batch[0]) * Count);
    }
    u16 getCount() const { return Count; }
    CompactPtrT get(u16 I) const {
      DCHECK_LE(I, Count);
      return Batch[I];
    }
    static u16 getMaxCached(uptr Size) {
      return Min(MaxNumCached, SizeClassMap::getMaxCachedHint(Size));
    }
    TransferBatch *Next;

  private:
    CompactPtrT Batch[MaxNumCached];
    u16 Count;
  };

  void init(GlobalStats *S, SizeClassAllocator *A) {
    DCHECK(isEmpty());
    Stats.init();
    if (LIKELY(S))
      S->link(&Stats);
    Allocator = A;
  }

  void destroy(GlobalStats *S) {
    drain();
    if (LIKELY(S))
      S->unlink(&Stats);
  }

  void *allocate(uptr ClassId) {
    DCHECK_LT(ClassId, NumClasses);
    PerClass *C = &PerClassArray[ClassId];
    if (C->Count == 0) {
      if (UNLIKELY(!refill(C, ClassId)))
        return nullptr;
      DCHECK_GT(C->Count, 0);
    }
    // We read ClassSize first before accessing Chunks because it's adjacent to
    // Count, while Chunks might be further off (depending on Count). That keeps
    // the memory accesses in close quarters.
    const uptr ClassSize = C->ClassSize;
    CompactPtrT CompactP = C->Chunks[--C->Count];
    Stats.add(StatAllocated, ClassSize);
    Stats.sub(StatFree, ClassSize);
    return Allocator->decompactPtr(ClassId, CompactP);
  }

  void deallocate(uptr ClassId, void *P) {
    CHECK_LT(ClassId, NumClasses);
    PerClass *C = &PerClassArray[ClassId];
    // We still have to initialize the cache in the event that the first heap
    // operation in a thread is a deallocation.
    initCacheMaybe(C);
    if (C->Count == C->MaxCount)
      drain(C, ClassId);
    // See comment in allocate() about memory accesses.
    const uptr ClassSize = C->ClassSize;
    C->Chunks[C->Count++] =
        Allocator->compactPtr(ClassId, reinterpret_cast<uptr>(P));
    Stats.sub(StatAllocated, ClassSize);
    Stats.add(StatFree, ClassSize);
  }

  bool isEmpty() const {
    for (uptr I = 0; I < NumClasses; ++I)
      if (PerClassArray[I].Count)
        return false;
    return true;
  }

  void drain() {
    // Drain BatchClassId last as createBatch can refill it.
    for (uptr I = 0; I < NumClasses; ++I) {
      if (I == BatchClassId)
        continue;
      while (PerClassArray[I].Count > 0)
        drain(&PerClassArray[I], I);
    }
    while (PerClassArray[BatchClassId].Count > 0)
      drain(&PerClassArray[BatchClassId], BatchClassId);
    DCHECK(isEmpty());
  }

  TransferBatch *createBatch(uptr ClassId, void *B) {
    if (ClassId != BatchClassId)
      B = allocate(BatchClassId);
    return reinterpret_cast<TransferBatch *>(B);
  }

  LocalStats &getStats() { return Stats; }

private:
  static const uptr NumClasses = SizeClassMap::NumClasses;
  static const uptr BatchClassId = SizeClassMap::BatchClassId;
  struct alignas(SCUDO_CACHE_LINE_SIZE) PerClass {
    u16 Count;
    u16 MaxCount;
    // Note: ClassSize is zero for the transfer batch.
    uptr ClassSize;
    CompactPtrT Chunks[2 * TransferBatch::MaxNumCached];
  };
  PerClass PerClassArray[NumClasses] = {};
  LocalStats Stats;
  SizeClassAllocator *Allocator = nullptr;

  ALWAYS_INLINE void initCacheMaybe(PerClass *C) {
    if (LIKELY(C->MaxCount))
      return;
    initCache();
    DCHECK_NE(C->MaxCount, 0U);
  }

  NOINLINE void initCache() {
    for (uptr I = 0; I < NumClasses; I++) {
      PerClass *P = &PerClassArray[I];
      const uptr Size = SizeClassAllocator::getSizeByClassId(I);
      P->MaxCount = 2 * TransferBatch::getMaxCached(Size);
      if (I != BatchClassId) {
        P->ClassSize = Size;
      } else {
        // ClassSize in this struct is only used for malloc/free stats, which
        // should only track user allocations, not internal movements.
        P->ClassSize = 0;
      }
    }
  }

  void destroyBatch(uptr ClassId, void *B) {
    if (ClassId != BatchClassId)
      deallocate(BatchClassId, B);
  }

  NOINLINE bool refill(PerClass *C, uptr ClassId) {
    initCacheMaybe(C);
    TransferBatch *B = Allocator->popBatch(this, ClassId);
    if (UNLIKELY(!B))
      return false;
    DCHECK_GT(B->getCount(), 0);
    C->Count = B->getCount();
    B->copyToArray(C->Chunks);
    B->clear();
    destroyBatch(ClassId, B);
    return true;
  }

  NOINLINE void drain(PerClass *C, uptr ClassId) {
    const u16 Count = Min(static_cast<u16>(C->MaxCount / 2), C->Count);
    TransferBatch *B =
        createBatch(ClassId, Allocator->decompactPtr(ClassId, C->Chunks[0]));
    if (UNLIKELY(!B))
      reportOutOfMemory(SizeClassAllocator::getSizeByClassId(BatchClassId));
    B->setFromArray(&C->Chunks[0], Count);
    C->Count -= Count;
    for (uptr I = 0; I < C->Count; I++)
      C->Chunks[I] = C->Chunks[I + Count];
    Allocator->pushBatch(ClassId, B);
  }
};

} // namespace scudo

#endif // SCUDO_LOCAL_CACHE_H_
