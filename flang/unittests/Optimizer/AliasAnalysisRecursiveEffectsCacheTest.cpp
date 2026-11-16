//===- AliasAnalysisRecursiveEffectsCacheTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for AliasAnalysisRecursiveEffectsCache, the opt-in cache that
// amortizes AliasAnalysis::getModRef queries against ops carrying
// HasRecursiveMemoryEffects (e.g. fir.do_loop). The cache is only consulted
// when an AliasAnalysis is constructed from one; it summarizes an op's nested
// effects once and then answers subsequent queries from that summary.
//
// The cache is a frozen snapshot with no automatic invalidation: a client
// enables it only across a region in which it does not mutate the summarized
// ops' bodies, and calls clear() otherwise.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct AliasAnalysisRecursiveEffectsCacheTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);
    builder = std::make_unique<mlir::OpBuilder>(&context);
    mlir::Location loc = builder->getUnknownLoc();

    moduleOp = mlir::ModuleOp::create(*builder, loc);
    builder->setInsertionPointToStart(moduleOp->getBody());
    mlir::func::FuncOp func = mlir::func::FuncOp::create(*builder, loc,
        "alias_analysis_recursive_cache_tests",
        builder->getFunctionType({}, {}));
    builder->setInsertionPointToStart(func.addEntryBlock());
  }

  mlir::Location getLoc() { return builder->getUnknownLoc(); }

  mlir::Value createAlloca() {
    return fir::AllocaOp::create(
        *builder, getLoc(), mlir::Float32Type::get(&context));
  }

  mlir::Value createIndex(std::int64_t v) {
    return mlir::arith::ConstantIndexOp::create(*builder, getLoc(), v);
  }

  /// Build `fir.do_loop %i = 0 to 10 step 1 { fir.store %cst to <dest> }` and
  /// return the loop op. fir.do_loop carries HasRecursiveMemoryEffects, so the
  /// cache summarizes the nested store rather than re-walking the body on
  /// every query.
  fir::DoLoopOp createLoopStoringTo(mlir::Value dest) {
    mlir::Value lb = createIndex(0);
    mlir::Value ub = createIndex(10);
    mlir::Value step = createIndex(1);
    auto loop = fir::DoLoopOp::create(*builder, getLoc(), lb, ub, step);

    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(loop.getBody());
    mlir::Value cst = mlir::arith::ConstantOp::create(
        *builder, getLoc(), builder->getF32FloatAttr(0.0f));
    fir::StoreOp::create(*builder, getLoc(), cst, dest);
    return loop;
  }

  mlir::MLIRContext context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
};

// An AliasAnalysis built without a cache never routes through one: the cache
// object stays empty and its counters stay at zero.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, UnusedWhenNotLinked) {
  mlir::Value a = createAlloca();
  fir::DoLoopOp loop = createLoopStoringTo(a);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa; // deliberately not constructed from `cache`

  (void)aa.getModRef(loop, a);
  (void)aa.getModRef(loop, a);

  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 0u);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 0u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 0u);
}

// The first query against a recursive op is a miss that builds and stores one
// summary; repeating it is a hit served from that summary, with no new entry.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, MissThenHit) {
  mlir::Value a = createAlloca();
  fir::DoLoopOp loop = createLoopStoringTo(a);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa{cache};

  (void)aa.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 0u);

  (void)aa.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 1u);

  // The summary is keyed on the op alone, not on the queried location, so a
  // different location against the same loop is also a hit.
  mlir::Value b = createAlloca();
  (void)aa.getModRef(loop, b);
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 2u);
}

// Distinct recursive ops are summarized independently.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, DistinctOpsGetDistinctEntries) {
  mlir::Value a = createAlloca();
  mlir::Value b = createAlloca();
  fir::DoLoopOp loopA = createLoopStoringTo(a);
  fir::DoLoopOp loopB = createLoopStoringTo(b);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa{cache};

  (void)aa.getModRef(loopA, a);
  (void)aa.getModRef(loopB, b);

  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 2u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 2u);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 0u);
}

// Non-recursive ops bypass the cache entirely: they are answered by the
// inline getModRef path and never summarized.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, NonRecursiveOpNotSummarized) {
  mlir::Value a = createAlloca();
  mlir::Value cst = mlir::arith::ConstantOp::create(
      *builder, getLoc(), builder->getF32FloatAttr(0.0f));
  auto store = fir::StoreOp::create(*builder, getLoc(), cst, a);
  ASSERT_FALSE(store->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>());

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa{cache};

  EXPECT_TRUE(aa.getModRef(store, a).isMod());
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 0u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 0u);
}

// clear() drops every summary, so the next query is a miss again. This is the
// escape hatch a client uses after mutating a summarized op's body.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, ClearDropsSummaries) {
  mlir::Value a = createAlloca();
  fir::DoLoopOp loop = createLoopStoringTo(a);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa{cache};

  (void)aa.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 1u);

  cache.clear();
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 0u);

  (void)aa.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheSizeForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 2u);
}

// The point of the cache is to be invisible: for every (op, location) pair it
// must produce exactly what an uncached AliasAnalysis produces.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, MatchesUncachedResult) {
  mlir::Value a = createAlloca();
  mlir::Value b = createAlloca();
  fir::DoLoopOp loopA = createLoopStoringTo(a);
  fir::DoLoopOp loopB = createLoopStoringTo(b);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis cachedAA{cache};
  fir::AliasAnalysis uncachedAA;

  for (mlir::Operation *op : {loopA.getOperation(), loopB.getOperation()}) {
    for (mlir::Value loc : {a, b}) {
      mlir::ModRefResult cached = cachedAA.getModRef(op, loc);
      mlir::ModRefResult uncached = uncachedAA.getModRef(op, loc);
      EXPECT_EQ(cached, uncached);
      // Query again to exercise the hit path, which must agree too.
      EXPECT_EQ(cachedAA.getModRef(op, loc), uncached);
    }
  }

  // A loop that only writes `a` modifies `a` and not `b`.
  EXPECT_TRUE(cachedAA.getModRef(loopA, a).isMod());
  EXPECT_TRUE(cachedAA.getModRef(loopA, b).isNoModRef());
}

// The cache holds a back-pointer to its AliasAnalysis, and AliasAnalysis holds
// one to the cache. Moving the analysis must re-link both, or the moved-to
// instance would silently stop caching (or, worse, the cache would delegate
// through a dangling pointer).
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, MoveKeepsCacheLinked) {
  mlir::Value a = createAlloca();
  fir::DoLoopOp loop = createLoopStoringTo(a);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis original{cache};
  (void)original.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 1u);

  fir::AliasAnalysis moved{std::move(original)};
  (void)moved.getModRef(loop, a);
  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 1u);
  EXPECT_EQ(cache.getSummaryCacheMissesForTesting(), 1u);
}

// The two caches are independent opt-ins serving different queries:
// enableSourceCache() memoizes getSource(), the recursive cache memoizes
// nested-effect summaries. LoopInvariantCodeMotion turns both on, so make sure
// they compose without interfering.
TEST_F(AliasAnalysisRecursiveEffectsCacheTest, ComposesWithSourceCache) {
  mlir::Value a = createAlloca();
  fir::DoLoopOp loop = createLoopStoringTo(a);

  fir::AliasAnalysisRecursiveEffectsCache cache;
  fir::AliasAnalysis aa{cache};
  aa.enableSourceCache();

  mlir::ModRefResult first = aa.getModRef(loop, a);
  mlir::ModRefResult second = aa.getModRef(loop, a);
  EXPECT_EQ(first, second);

  EXPECT_EQ(cache.getSummaryCacheHitsForTesting(), 1u);
  // The summary hit still resolves locations through alias(), which goes
  // through getSource(), so the source cache is exercised as well.
  EXPECT_GT(aa.getSourceCacheSizeForTesting(), 0u);
}
