//===- AliasAnalysisCacheTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the opt-in getSource() memoization cache on
// fir::AliasAnalysis. The cache is off by default; when enabled it memoizes
// getSource() results for the lifetime of the analysis instance and is a
// frozen snapshot (no automatic invalidation).
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct AliasAnalysisCacheTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);
    builder = std::make_unique<mlir::OpBuilder>(&context);
    mlir::Location loc = builder->getUnknownLoc();

    // Set up a module with a dummy function; insert into its entry block.
    moduleOp = mlir::ModuleOp::create(*builder, loc);
    builder->setInsertionPointToStart(moduleOp->getBody());
    mlir::func::FuncOp func = mlir::func::FuncOp::create(*builder, loc,
        "alias_analysis_cache_tests", builder->getFunctionType({}, {}));
    auto *entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
  }

  mlir::Location getLoc() { return builder->getUnknownLoc(); }

  mlir::Value createAlloca() {
    return fir::AllocaOp::create(
        *builder, getLoc(), mlir::Float32Type::get(&context));
  }

  mlir::MLIRContext context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
};

// Caching is off by default: getSource() bypasses the cache, so nothing is
// memoized and the hit/miss counters stay at zero.
TEST_F(AliasAnalysisCacheTest, DisabledByDefault) {
  mlir::Value a = createAlloca();
  fir::AliasAnalysis aa;

  (void)aa.getSource(a);
  (void)aa.getSource(a);

  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);
  EXPECT_EQ(aa.getSourceCacheHitsForTesting(), 0u);
  EXPECT_EQ(aa.getSourceCacheMissesForTesting(), 0u);
}

// Once enabled, a query populates the cache (a miss) and a repeated query for
// the same value is served from it (a hit) without growing the cache.
TEST_F(AliasAnalysisCacheTest, FillAndHit) {
  mlir::Value a = createAlloca();
  mlir::Value b = createAlloca();
  fir::AliasAnalysis aa;
  aa.enableSourceCache();

  (void)aa.getSource(a);
  std::size_t sizeAfterFill = aa.getSourceCacheSizeForTesting();
  std::size_t missesAfterFill = aa.getSourceCacheMissesForTesting();
  EXPECT_GE(sizeAfterFill, 1u);
  EXPECT_GE(missesAfterFill, 1u);
  EXPECT_EQ(aa.getSourceCacheHitsForTesting(), 0u);

  // Repeating the top-level query is a single cache hit: it returns the
  // memoized result without recomputing (and thus without new misses or
  // entries), regardless of any recursive sub-queries the first call made.
  (void)aa.getSource(a);
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), sizeAfterFill);
  EXPECT_EQ(aa.getSourceCacheMissesForTesting(), missesAfterFill);
  EXPECT_EQ(aa.getSourceCacheHitsForTesting(), 1u);

  // A distinct value adds at least one new entry.
  (void)aa.getSource(b);
  EXPECT_GT(aa.getSourceCacheSizeForTesting(), sizeAfterFill);
  EXPECT_GT(aa.getSourceCacheMissesForTesting(), missesAfterFill);
}

// disableSourceCache() clears the entries and turns memoization off, so later
// queries bypass the cache and leave it empty.
TEST_F(AliasAnalysisCacheTest, DisableClearsAndBypasses) {
  mlir::Value a = createAlloca();
  fir::AliasAnalysis aa;
  aa.enableSourceCache();

  (void)aa.getSource(a);
  EXPECT_GE(aa.getSourceCacheSizeForTesting(), 1u);

  aa.disableSourceCache();
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);

  (void)aa.getSource(a);
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);
}
