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
#include "report.h"
#include "stats.h"

namespace scudo {

template <class SizeClassAllocator> struct SizeClassAllocatorLocalCache {
  typedef typename SizeClassAllocator::SizeClassMap SizeClassMap;

  struct TransferBatch {
    static const u32 MaxNumCached = SizeClassMap::MaxNumCachedHint;
    void setFromArray(void **Array, u32 N) {
      DCHECK_LE(N, MaxNumCached);
      Count = N;
      memcpy(Batch, Array, sizeof(void *) * Count);
    }
    void clear() { Count = 0; }
    void add(void *P) {
      DCHECK_LT(Count, MaxNumCached);
      Batch[Count++] = P;
    }
    void copyToArray(void **Array) const {
      memcpy(Array, Batch, sizeof(void *) * Count);
    }
    u32 getCount() const { return Count; }
    void *get(u32 I) const {
      DCHECK_LE(I, Count);
      return Batch[I];
    }
    static u32 getMaxCached(uptr Size) {
      return Min(MaxNumCached, SizeClassMap::getMaxCachedHint(Size));
    }
    TransferBatch *Next;

  private:
    u32 Count;
    void *Batch[MaxNumCached];
  };

  void initLinkerInitialized(GlobalStats *S, SizeClassAllocator *A) {
    Stats.initLinkerInitialized();
    if (LIKELY(S))
      S->link(&Stats);
    Allocator = A;
  }

  void init(GlobalStats *S, SizeClassAllocator *A) {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized(S, A);
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
    void *P = C->Chunks[--C->Count];
    // The jury is still out as to whether any kind of PREFETCH here increases
    // performance. It definitely decreases performance on Android though.
    // if (!SCUDO_ANDROID) PREFETCH(P);
    Stats.add(StatAllocated, ClassSize);
    Stats.sub(StatFree, ClassSize);
    return P;
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
    C->Chunks[C->Count++] = P;
    Stats.sub(StatAllocated, ClassSize);
    Stats.add(StatFree, ClassSize);
  }

  void drain() {
    for (uptr I = 0; I < NumClasses; I++) {
      PerClass *C = &PerClassArray[I];
      while (C->Count > 0)
        drain(C, I);
    }
  }

  TransferBatch *createBatch(uptr ClassId, void *B) {
    if (ClassId != SizeClassMap::BatchClassId)
      B = allocate(SizeClassMap::BatchClassId);
    return reinterpret_cast<TransferBatch *>(B);
  }

  LocalStats &getStats() { return Stats; }

private:
  static const uptr NumClasses = SizeClassMap::NumClasses;
  struct PerClass {
    u32 Count;
    u32 MaxCount;
    uptr ClassSize;
    void *Chunks[2 * TransferBatch::MaxNumCached];
  };
  PerClass PerClassArray[NumClasses];
  LocalStats Stats;
  SizeClassAllocator *Allocator;

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
      P->ClassSize = Size;
    }
  }

  void destroyBatch(uptr ClassId, void *B) {
    if (ClassId != SizeClassMap::BatchClassId)
      deallocate(SizeClassMap::BatchClassId, B);
  }

  NOINLINE bool refill(PerClass *C, uptr ClassId) {
    initCacheMaybe(C);
    TransferBatch *B = Allocator->popBatch(this, ClassId);
    if (UNLIKELY(!B))
      return false;
    DCHECK_GT(B->getCount(), 0);
    C->Count = B->getCount();
    B->copyToArray(C->Chunks);
    destroyBatch(ClassId, B);
    return true;
  }

  NOINLINE void drain(PerClass *C, uptr ClassId) {
    const u32 Count = Min(C->MaxCount / 2, C->Count);
    TransferBatch *B = createBatch(ClassId, C->Chunks[0]);
    if (UNLIKELY(!B))
      reportOutOfMemory(
          SizeClassAllocator::getSizeByClassId(SizeClassMap::BatchClassId));
    B->setFromArray(&C->Chunks[0], Count);
    C->Count -= Count;
    for (uptr I = 0; I < C->Count; I++)
      C->Chunks[I] = C->Chunks[I + Count];
    Allocator->pushBatch(ClassId, B);
  }
};

} // namespace scudo

#endif // SCUDO_LOCAL_CACHE_H_
