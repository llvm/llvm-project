//===- AliasAnalysisCacheTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the listener-driven invalidation of fir::AliasAnalysis's getSource
// cache: mutating the IR through a rewriter must evict precisely the cached
// entries that depend on the mutated operation, and an erase must drop the
// affected entries before the operation's storage is freed (so a freed,
// possibly reused, pointer can never produce a stale cache hit).
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct AliasAnalysisCacheTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);
    builder = std::make_unique<mlir::OpBuilder>(&context);
    mlir::Location loc = builder->getUnknownLoc();

    moduleOp = mlir::ModuleOp::create(*builder, loc);
    builder->setInsertionPointToStart(moduleOp->getBody());
    mlir::func::FuncOp func = mlir::func::FuncOp::create(
        *builder, loc, "test", builder->getFunctionType({}, {}));
    builder->setInsertionPointToStart(func.addEntryBlock());
  }

  mlir::Location getLoc() { return builder->getUnknownLoc(); }

  // Build an `alloca -> declare` chain for a scalar i32 variable and return the
  // declared variable address.
  mlir::Value createScalarVariable(llvm::StringRef name) {
    mlir::Location loc = getLoc();
    mlir::Type eleType = mlir::IntegerType::get(&context, 32);
    mlir::Value addr = fir::AllocaOp::create(*builder, loc, eleType);
    auto declare = fir::DeclareOp::create(*builder, loc, addr.getType(), addr,
        /*shape=*/mlir::Value{}, /*typeParams=*/mlir::ValueRange{},
        /*dummy_scope=*/nullptr, /*storage=*/nullptr, /*storage_offset=*/0,
        mlir::StringAttr::get(&context, name),
        /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
        /*data_attr=*/cuf::DataAttributeAttr{},
        /*dummy_arg_no=*/mlir::IntegerAttr{});
    return declare.getResult();
  }

  mlir::MLIRContext context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
};

// Modifying an operation that a cached query transitively depends on (here the
// alloca underlying a declared variable) must evict that query's entry while
// leaving unrelated entries intact.
TEST_F(AliasAnalysisCacheTest, PreciseEvictionOnModify) {
  mlir::Value var1 = createScalarVariable("x1");
  mlir::Value var2 = createScalarVariable("x2");

  fir::AliasAnalysis aa;
  mlir::IRRewriter rewriter(&context);
  aa.enableSourceCache(&rewriter);

  fir::AliasAnalysis::Source s1 = aa.getSource(var1);
  fir::AliasAnalysis::Source s2 = aa.getSource(var2);
  std::size_t before = aa.getSourceCacheSizeForTesting();
  EXPECT_GE(before, 2u);

  // var1's source depends on its alloca; modifying the alloca must evict the
  // var1 entry (a transitive dependency, not the queried op itself).
  mlir::Operation *alloca1 =
      var1.getDefiningOp()->getOperand(0).getDefiningOp();
  ASSERT_TRUE(alloca1);
  rewriter.modifyOpInPlace(alloca1, [] {});

  std::size_t after = aa.getSourceCacheSizeForTesting();
  EXPECT_LT(after, before); // some entry was evicted
  EXPECT_GT(after, 0u); // but not everything (var2 is unaffected)

  // Re-querying var2 must still return the same source (it was not evicted).
  fir::AliasAnalysis::Source s2again = aa.getSource(var2);
  EXPECT_EQ(s2again.origin, s2.origin);
  EXPECT_EQ(s2again.kind, s2.kind);

  // Sanity: var1 and var2 resolve to distinct origins.
  EXPECT_NE(s1.origin, s2.origin);

  aa.disableSourceCache();
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);
}

// Erasing a cached query's defining operation must drop the affected entries
// (the notification fires before the storage is freed), so no stale entry can
// survive to alias with a later, pointer-reused operation.
TEST_F(AliasAnalysisCacheTest, EvictionOnEraseClosesReuseHole) {
  mlir::Value var1 = createScalarVariable("x1");
  mlir::Value var2 = createScalarVariable("x2");

  // A load whose result we query; it has no uses, so it can be erased.
  mlir::Value load1 = fir::LoadOp::create(*builder, getLoc(), var1);

  fir::AliasAnalysis aa;
  mlir::IRRewriter rewriter(&context);
  aa.enableSourceCache(&rewriter);

  aa.getSource(load1);
  aa.getSource(var2);
  std::size_t before = aa.getSourceCacheSizeForTesting();
  EXPECT_GE(before, 2u);

  // Erase the load: its cache entry must be gone afterwards, while var2's
  // entry survives.
  mlir::Operation *loadOp = load1.getDefiningOp();
  rewriter.eraseOp(loadOp);

  std::size_t after = aa.getSourceCacheSizeForTesting();
  EXPECT_LT(after, before);
  EXPECT_GT(after, 0u);

  // var2 is still cached and valid.
  fir::AliasAnalysis::Source s2 = aa.getSource(var2);
  EXPECT_EQ(s2.kind, fir::AliasAnalysis::SourceKind::Allocate);

  aa.disableSourceCache();
}

// Without a rewriter the cache is a frozen snapshot: it still memoizes results,
// but installs no listener (so the caller is responsible for scoping).
TEST_F(AliasAnalysisCacheTest, FrozenModeMemoizes) {
  mlir::Value var1 = createScalarVariable("x1");

  fir::AliasAnalysis aa;
  aa.enableSourceCache(/*rewriter=*/nullptr);
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);

  fir::AliasAnalysis::Source first = aa.getSource(var1);
  EXPECT_GT(aa.getSourceCacheSizeForTesting(), 0u);
  fir::AliasAnalysis::Source second = aa.getSource(var1);
  EXPECT_EQ(first.origin, second.origin);

  aa.disableSourceCache();
  EXPECT_EQ(aa.getSourceCacheSizeForTesting(), 0u);
}
