//===-- quarantine_test.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "quarantine.h"

#include <pthread.h>
#include <stdlib.h>

#include "tests/scudo_unit_test.h"

namespace {

void *FakePtr = reinterpret_cast<void *>(0xFA83FA83);
const scudo::uptr BlockSize = 8UL;
const scudo::uptr LargeBlockSize = 16384UL;

struct QuarantineCallback {
  void recycle(void *P) { EXPECT_EQ(P, FakePtr); }
  void *allocate(scudo::uptr Size) { return malloc(Size); }
  void deallocate(void *P) { free(P); }
};

typedef scudo::GlobalQuarantine<QuarantineCallback, void> QuarantineT;
typedef typename QuarantineT::CacheT CacheT;

QuarantineCallback Cb;

void deallocateCache(CacheT *Cache) {
  while (scudo::QuarantineBatch *Batch = Cache->dequeueBatch())
    Cb.deallocate(Batch);
}

TEST(ScudoQuarantineTest, QuarantineBatchMerge) {
  // Verify the trivial case.
  scudo::QuarantineBatch Into;
  Into.init(FakePtr, 4UL);
  scudo::QuarantineBatch From;
  From.init(FakePtr, 8UL);

  Into.merge(&From);

  EXPECT_EQ(Into.Count, 2UL);
  EXPECT_EQ(Into.Batch[0], FakePtr);
  EXPECT_EQ(Into.Batch[1], FakePtr);
  EXPECT_EQ(Into.Size, 12UL + sizeof(scudo::QuarantineBatch));
  EXPECT_EQ(Into.getQuarantinedSize(), 12UL);

  EXPECT_EQ(From.Count, 0UL);
  EXPECT_EQ(From.Size, sizeof(scudo::QuarantineBatch));
  EXPECT_EQ(From.getQuarantinedSize(), 0UL);

  // Merge the batch to the limit.
  for (scudo::uptr I = 2; I < scudo::QuarantineBatch::MaxCount; ++I)
    From.push_back(FakePtr, 8UL);
  EXPECT_TRUE(Into.Count + From.Count == scudo::QuarantineBatch::MaxCount);
  EXPECT_TRUE(Into.canMerge(&From));

  Into.merge(&From);
  EXPECT_TRUE(Into.Count == scudo::QuarantineBatch::MaxCount);

  // No more space, not even for one element.
  From.init(FakePtr, 8UL);

  EXPECT_FALSE(Into.canMerge(&From));
}

TEST(ScudoQuarantineTest, QuarantineBatchBounds) {
  scudo::QuarantineBatch Into;
  scudo::QuarantineBatch From;
  Into.init(FakePtr, BlockSize);
  From.init(FakePtr, BlockSize);
  Into.merge(&From);
  // Empty batches produced by merging remain valid for merging and shuffling.
  EXPECT_TRUE(Into.canMerge(&From));
  Into.merge(&From);
  From.shuffle(1);
  EXPECT_EQ(From.Count, 0U);

  while (Into.Count < scudo::QuarantineBatch::MaxCount)
    Into.push_back(FakePtr, BlockSize);
  EXPECT_TRUE(Into.canMerge(&From));
  Into.merge(&From);
  Into.shuffle(1);
  EXPECT_TRUE(Into.Count == scudo::QuarantineBatch::MaxCount);
  for (scudo::u32 I = 0; I < Into.Count; ++I)
    EXPECT_EQ(Into.Batch[I], FakePtr);
}

TEST(ScudoQuarantineDeathTest, QuarantineBatchPushBackFull) {
  scudo::QuarantineBatch B;
  B.init(FakePtr, BlockSize);
  while (B.Count < scudo::QuarantineBatch::MaxCount)
    B.push_back(FakePtr, BlockSize);
  SCUDO_EXPECT_DEATH(B.push_back(FakePtr, BlockSize), "Count");
}

TEST(ScudoQuarantineDeathTest, QuarantineBatchInvalidCount) {
  for (scudo::u32 Count : {scudo::QuarantineBatch::MaxCount + 1, UINT32_MAX}) {
    scudo::QuarantineBatch B;
    scudo::QuarantineBatch Other;
    B.init(FakePtr, BlockSize);
    Other.init(FakePtr, BlockSize);
    B.Count = Count;
    SCUDO_EXPECT_DEATH(B.push_back(FakePtr, BlockSize), "Count");
    SCUDO_EXPECT_DEATH(B.canMerge(&Other), "Count");
    SCUDO_EXPECT_DEATH(Other.canMerge(&B), "Count");
    SCUDO_EXPECT_DEATH(B.merge(&Other), "Count");
    SCUDO_EXPECT_DEATH(Other.merge(&B), "Count");
    SCUDO_EXPECT_DEATH(B.shuffle(1), "Count");
  }
}

TEST(ScudoQuarantineDeathTest, QuarantineBatchMergeOverflow) {
  scudo::QuarantineBatch Into;
  scudo::QuarantineBatch From;
  Into.init(FakePtr, BlockSize);
  From.init(FakePtr, BlockSize);
  Into.Count = scudo::QuarantineBatch::MaxCount;
  EXPECT_FALSE(Into.canMerge(&From));
  SCUDO_EXPECT_DEATH(Into.merge(&From), "canMerge");

  // A sum of corrupted counts must not wrap around and appear to fit.
  Into.Count = From.Count = 1U << 31;
  SCUDO_EXPECT_DEATH(Into.canMerge(&From), "Count");
  SCUDO_EXPECT_DEATH(Into.merge(&From), "Count");
}

TEST(ScudoQuarantineDeathTest, QuarantineCacheEnqueueInvalidCount) {
  CacheT Cache;
  Cache.init();
  scudo::QuarantineBatch B;
  B.init(FakePtr, BlockSize);
  Cache.enqueueBatch(&B);
  // An oversized count bypasses enqueue's full-batch equality check.
  B.Count = scudo::QuarantineBatch::MaxCount + 1;
  SCUDO_EXPECT_DEATH(Cache.enqueue(Cb, FakePtr, BlockSize), "Count");
  EXPECT_EQ(Cache.dequeueBatch(), &B);
}

TEST(ScudoQuarantineDeathTest, GlobalQuarantineRecycleInvalidCount) {
  QuarantineT Quarantine;
  CacheT Cache;
  Cache.init();
  Quarantine.init(1024UL << 10, 256UL << 10);
  auto *B = static_cast<scudo::QuarantineBatch *>(
      Cb.allocate(sizeof(scudo::QuarantineBatch)));
  ASSERT_NE(B, nullptr);
  B->init(FakePtr, BlockSize);
  Cache.enqueueBatch(B);
  B->Count = scudo::QuarantineBatch::MaxCount + 1;
  SCUDO_EXPECT_DEATH(Quarantine.drainAndRecycle(&Cache, Cb), "Count");
  deallocateCache(&Cache);
}

TEST(ScudoQuarantineTest, QuarantineCacheMergeBatchesEmpty) {
  CacheT Cache;
  CacheT ToDeallocate;
  Cache.init();
  ToDeallocate.init();
  Cache.mergeBatches(&ToDeallocate);

  EXPECT_EQ(ToDeallocate.getSize(), 0UL);
  EXPECT_EQ(ToDeallocate.dequeueBatch(), nullptr);
}

TEST(SanitizerCommon, QuarantineCacheMergeBatchesOneBatch) {
  CacheT Cache;
  Cache.init();
  Cache.enqueue(Cb, FakePtr, BlockSize);
  EXPECT_EQ(BlockSize + sizeof(scudo::QuarantineBatch), Cache.getSize());

  CacheT ToDeallocate;
  ToDeallocate.init();
  Cache.mergeBatches(&ToDeallocate);

  // Nothing to merge, nothing to deallocate.
  EXPECT_EQ(BlockSize + sizeof(scudo::QuarantineBatch), Cache.getSize());

  EXPECT_EQ(ToDeallocate.getSize(), 0UL);
  EXPECT_EQ(ToDeallocate.dequeueBatch(), nullptr);

  deallocateCache(&Cache);
}

TEST(ScudoQuarantineTest, QuarantineCacheMergeBatchesSmallBatches) {
  // Make a Cache with two batches small enough to merge.
  CacheT From;
  From.init();
  From.enqueue(Cb, FakePtr, BlockSize);
  CacheT Cache;
  Cache.init();
  Cache.enqueue(Cb, FakePtr, BlockSize);

  Cache.transfer(&From);
  EXPECT_EQ(BlockSize * 2 + sizeof(scudo::QuarantineBatch) * 2,
            Cache.getSize());

  CacheT ToDeallocate;
  ToDeallocate.init();
  Cache.mergeBatches(&ToDeallocate);

  // Batches merged, one batch to deallocate.
  EXPECT_EQ(BlockSize * 2 + sizeof(scudo::QuarantineBatch), Cache.getSize());
  EXPECT_EQ(ToDeallocate.getSize(), sizeof(scudo::QuarantineBatch));

  deallocateCache(&Cache);
  deallocateCache(&ToDeallocate);
}

TEST(ScudoQuarantineTest, QuarantineCacheMergeBatchesTooBigToMerge) {
  const scudo::uptr NumBlocks = scudo::QuarantineBatch::MaxCount - 1;

  // Make a Cache with two batches small enough to merge.
  CacheT From;
  CacheT Cache;
  From.init();
  Cache.init();
  for (scudo::uptr I = 0; I < NumBlocks; ++I) {
    From.enqueue(Cb, FakePtr, BlockSize);
    Cache.enqueue(Cb, FakePtr, BlockSize);
  }
  Cache.transfer(&From);
  EXPECT_EQ(BlockSize * NumBlocks * 2 + sizeof(scudo::QuarantineBatch) * 2,
            Cache.getSize());

  CacheT ToDeallocate;
  ToDeallocate.init();
  Cache.mergeBatches(&ToDeallocate);

  // Batches cannot be merged.
  EXPECT_EQ(BlockSize * NumBlocks * 2 + sizeof(scudo::QuarantineBatch) * 2,
            Cache.getSize());
  EXPECT_EQ(ToDeallocate.getSize(), 0UL);

  deallocateCache(&Cache);
}

TEST(ScudoQuarantineTest, QuarantineCacheMergeBatchesALotOfBatches) {
  const scudo::uptr NumBatchesAfterMerge = 3;
  const scudo::uptr NumBlocks =
      scudo::QuarantineBatch::MaxCount * NumBatchesAfterMerge;
  const scudo::uptr NumBatchesBeforeMerge = NumBlocks;

  // Make a Cache with many small batches.
  CacheT Cache;
  Cache.init();
  for (scudo::uptr I = 0; I < NumBlocks; ++I) {
    CacheT From;
    From.init();
    From.enqueue(Cb, FakePtr, BlockSize);
    Cache.transfer(&From);
  }

  EXPECT_EQ(BlockSize * NumBlocks +
                sizeof(scudo::QuarantineBatch) * NumBatchesBeforeMerge,
            Cache.getSize());

  CacheT ToDeallocate;
  ToDeallocate.init();
  Cache.mergeBatches(&ToDeallocate);

  // All blocks should fit Into 3 batches.
  EXPECT_EQ(BlockSize * NumBlocks +
                sizeof(scudo::QuarantineBatch) * NumBatchesAfterMerge,
            Cache.getSize());

  EXPECT_EQ(ToDeallocate.getSize(),
            sizeof(scudo::QuarantineBatch) *
                (NumBatchesBeforeMerge - NumBatchesAfterMerge));

  deallocateCache(&Cache);
  deallocateCache(&ToDeallocate);
}

const scudo::uptr MaxQuarantineSize = 1024UL << 10; // 1MB
const scudo::uptr MaxCacheSize = 256UL << 10;       // 256KB

TEST(ScudoQuarantineTest, GlobalQuarantine) {
  QuarantineT Quarantine;
  CacheT Cache;
  Cache.init();
  Quarantine.init(MaxQuarantineSize, MaxCacheSize);
  EXPECT_EQ(Quarantine.getMaxSize(), MaxQuarantineSize);
  EXPECT_EQ(Quarantine.getCacheSize(), MaxCacheSize);

  bool DrainOccurred = false;
  scudo::uptr CacheSize = Cache.getSize();
  EXPECT_EQ(Cache.getSize(), 0UL);
  // We quarantine enough blocks that a drain has to occur. Verify this by
  // looking for a decrease of the size of the cache.
  for (scudo::uptr I = 0; I < 128UL; I++) {
    Quarantine.put(&Cache, Cb, FakePtr, LargeBlockSize);
    if (!DrainOccurred && Cache.getSize() < CacheSize)
      DrainOccurred = true;
    CacheSize = Cache.getSize();
  }
  EXPECT_TRUE(DrainOccurred);

  Quarantine.drainAndRecycle(&Cache, Cb);
  EXPECT_EQ(Cache.getSize(), 0UL);

  if (TEST_HAS_FAILURE) {
    scudo::ScopedString Str;
    Quarantine.getStats(&Str);
    Str.output();
  }
}

struct PopulateQuarantineThread {
  pthread_t Thread;
  QuarantineT *Quarantine;
  CacheT Cache;
};

void *populateQuarantine(void *Param) {
  PopulateQuarantineThread *P = static_cast<PopulateQuarantineThread *>(Param);
  P->Cache.init();
  for (scudo::uptr I = 0; I < 128UL; I++)
    P->Quarantine->put(&P->Cache, Cb, FakePtr, LargeBlockSize);
  return 0;
}

TEST(ScudoQuarantineTest, ThreadedGlobalQuarantine) {
  QuarantineT Quarantine;
  Quarantine.init(MaxQuarantineSize, MaxCacheSize);

  const scudo::uptr NumberOfThreads = 32U;
  PopulateQuarantineThread T[NumberOfThreads];
  for (scudo::uptr I = 0; I < NumberOfThreads; I++) {
    T[I].Quarantine = &Quarantine;
    pthread_create(&T[I].Thread, 0, populateQuarantine, &T[I]);
  }
  for (scudo::uptr I = 0; I < NumberOfThreads; I++)
    pthread_join(T[I].Thread, 0);

  if (TEST_HAS_FAILURE) {
    scudo::ScopedString Str;
    Quarantine.getStats(&Str);
    Str.output();
  }

  for (scudo::uptr I = 0; I < NumberOfThreads; I++)
    Quarantine.drainAndRecycle(&T[I].Cache, Cb);
}

} // namespace
