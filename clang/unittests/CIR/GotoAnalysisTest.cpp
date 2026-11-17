//===- GotoAnalysisTest.cpp - Tests for cir::GotoAnalysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/GotoAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {

class GotoAnalysisTest : public ::testing::Test {
protected:
  GotoAnalysisTest() { context.loadDialect<cir::CIRDialect>(); }

  OwningOpRef<ModuleOp> parse(StringRef body) {
    std::string ir = "!s32i = !cir.int<s, 32>\n" + body.str();
    return parseSourceString<ModuleOp>(ir, &context);
  }

  /// Return the `cir.const` holding `value`. The tests mark positions with
  /// constants of distinct values.
  static Operation *getMarker(ModuleOp module, int64_t value) {
    Operation *marker = nullptr;
    module.walk([&](cir::ConstantOp op) {
      if (auto attr = dyn_cast<cir::IntAttr>(op.getValue());
          attr && attr.getSInt() == value)
        marker = op;
    });
    return marker;
  }

  static cir::FuncOp getFunc(ModuleOp module, unsigned index = 0) {
    return *std::next(module.getOps<cir::FuncOp>().begin(), index);
  }

  /// Expect `analysis`, kept up to date through rewrites, to agree with a
  /// fresh analysis on every operation.
  static void expectFresh(cir::GotoAnalysis &analysis, ModuleOp module) {
    cir::GotoAnalysis fresh(module);
    module.walk([&](Operation *op) {
      if (op != module.getOperation())
        EXPECT_EQ(analysis.isReachable(op), fresh.isReachable(op))
            << "on " << op->getName().getStringRef().str();
    });
  }

  MLIRContext context;
};

TEST_F(GotoAnalysisTest, LabelHolder) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.goto "in"
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.scope {
        %1 = cir.const #cir.int<2> : !s32i
        cir.br ^bb2
      ^bb2:
        cir.label "in"
        %2 = cir.const #cir.int<3> : !s32i
        cir.yield
      }
      %3 = cir.const #cir.int<4> : !s32i
      cir.return
    ^bb3:
      %4 = cir.const #cir.int<5> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  // Before the scope that holds the label.
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 1)));
  // In the entry block of the holder, which counts as entered.
  EXPECT_TRUE(analysis.isReachable(getMarker(*module, 2)));
  // After the label, and after its holder.
  EXPECT_TRUE(analysis.isReachable(getMarker(*module, 3)));
  EXPECT_TRUE(analysis.isReachable(getMarker(*module, 4)));
  // In a block that nothing reaches.
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 5)));
}

TEST_F(GotoAnalysisTest, NestedCase) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%arg0: !s32i) {
      cir.return
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.switch(%arg0 : !s32i) {
        cir.case(equal, [#cir.int<1> : !s32i]) {
          cir.scope {
            cir.break
          ^bb1:
            %1 = cir.const #cir.int<2> : !s32i
            cir.case(equal, [#cir.int<2> : !s32i]) {
              %2 = cir.const #cir.int<3> : !s32i
              cir.yield
            }
            %3 = cir.const #cir.int<4> : !s32i
            cir.yield
          }
          cir.yield
        }
        cir.yield
      }
      %4 = cir.const #cir.int<5> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  // Only the switch jumps to its cases, and nothing reaches the switch here.
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 1)));
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 3)));
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 5)));

  // Once an edge reaches the switch, the switch reaches its nested case, but
  // not the constant before the case.
  OpBuilder builder(&context);
  cir::FuncOp func = getFunc(*module);
  Operation *ret = func.getBody().front().getTerminator();
  builder.setInsertionPoint(ret);
  cir::BrOp::create(builder, ret->getLoc(),
                    &*std::next(func.getBody().begin()));
  ret->erase();
  cir::GotoAnalysis afterEdge(*module);
  EXPECT_FALSE(afterEdge.isReachable(getMarker(*module, 2)));
  EXPECT_TRUE(afterEdge.isReachable(getMarker(*module, 3)));
  EXPECT_TRUE(afterEdge.isReachable(getMarker(*module, 4)));
  EXPECT_TRUE(afterEdge.isReachable(getMarker(*module, 5)));
}

TEST_F(GotoAnalysisTest, LabelInIsolatedOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.return
    ^bb1:
      cir.scope {
        builtin.module {
          cir.func @g() {
            cir.goto "in"
          ^bb1:
            cir.label "in"
            cir.return
          }
        }
        cir.yield
      }
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  // A goto in @g cannot enter @f, so the scope does not hold a jump target.
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 1)));
}

TEST_F(GotoAnalysisTest, AddedEdge) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.return
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_FALSE(analysis.isReachable(marker));

  // Branch to the dead block through a rewriter that updates the analysis.
  IRRewriter rewriter(&context, &analysis);
  Operation *ret = getFunc(*module).getBody().front().getTerminator();
  rewriter.setInsertionPoint(ret);
  cir::BrOp::create(rewriter, ret->getLoc(), marker->getBlock());
  rewriter.eraseOp(ret);
  EXPECT_TRUE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, MergeAndLostEdge) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%arg0: !cir.bool) {
      cir.brcond %arg0 ^bb1, ^bb3
    ^bb1:
      cir.br ^bb2
    ^bb2:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    ^bb3:
      %1 = cir.const #cir.int<2> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *afterMerge = getMarker(*module, 1);
  Operation *afterLostEdge = getMarker(*module, 2);
  EXPECT_TRUE(analysis.isReachable(afterMerge));
  EXPECT_TRUE(analysis.isReachable(afterLostEdge));

  IRRewriter rewriter(&context, &analysis);
  Block *entry = &getFunc(*module).getBody().front();

  // Merging ^bb2 into ^bb1 keeps every remaining block reachable.
  Block *bb1 = entry->getSuccessor(0);
  Block *bb2 = bb1->getSuccessor(0);
  rewriter.eraseOp(bb1->getTerminator());
  rewriter.mergeBlocks(bb2, bb1);
  EXPECT_TRUE(analysis.isReachable(afterMerge));
  EXPECT_TRUE(analysis.isReachable(afterLostEdge));
  expectFresh(analysis, *module);

  // Dropping the edge to ^bb3 disconnects it.
  Operation *brcond = entry->getTerminator();
  rewriter.setInsertionPoint(brcond);
  cir::BrOp::create(rewriter, brcond->getLoc(), bb1);
  rewriter.eraseOp(brcond);
  EXPECT_TRUE(analysis.isReachable(afterMerge));
  EXPECT_FALSE(analysis.isReachable(afterLostEdge));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, ErasedBlock) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.br ^bb1
    ^bb1:
      cir.br ^bb2
    ^bb2:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_TRUE(analysis.isReachable(marker));

  // Erasing ^bb1 is not a merge: nothing takes over its edge to ^bb2.
  IRRewriter rewriter(&context, &analysis);
  Block *entry = &getFunc(*module).getBody().front();
  Block *bb1 = entry->getSuccessor(0);
  Operation *br = entry->getTerminator();
  rewriter.setInsertionPoint(br);
  cir::ReturnOp::create(rewriter, br->getLoc());
  rewriter.eraseOp(br);
  rewriter.eraseBlock(bb1);
  EXPECT_FALSE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, EdgeIntoHolderBlock) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.goto "in"
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.scope {
        cir.br ^bb2
      ^bb2:
        cir.label "in"
        cir.yield
      }
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_FALSE(analysis.isReachable(marker));

  // An edge into the block reaches it from its start, not from the holder on.
  IRRewriter rewriter(&context, &analysis);
  Operation *jump = getFunc(*module).getBody().front().getTerminator();
  rewriter.setInsertionPoint(jump);
  cir::BrOp::create(rewriter, jump->getLoc(), marker->getBlock());
  rewriter.eraseOp(jump);
  EXPECT_TRUE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, MovedHolder) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.goto "in"
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.scope {
        cir.br ^bb2
      ^bb2:
        cir.label "in"
        cir.yield
      }
      cir.return
    ^bb3:
      %1 = cir.const #cir.int<2> : !s32i
      %2 = cir.const #cir.int<3> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 3)));

  // Control now enters ^bb3 at the moved holder.
  IRRewriter rewriter(&context, &analysis);
  Operation *holder = getMarker(*module, 1)->getNextNode();
  rewriter.moveOpBefore(holder, getMarker(*module, 3));
  EXPECT_FALSE(analysis.isReachable(getMarker(*module, 2)));
  EXPECT_TRUE(analysis.isReachable(getMarker(*module, 3)));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, LabelLeavesLoop) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%arg0: !cir.bool) {
      cir.return
    ^bb1:
      cir.label "in"
      %0 = cir.const #cir.int<1> : !s32i
      cir.brcond %arg0 ^bb1, ^bb2
    ^bb2:
      %1 = cir.const #cir.int<2> : !s32i
      cir.return
    ^bb3:
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *inLoop = getMarker(*module, 1);
  Operation *afterLoop = getMarker(*module, 2);
  EXPECT_TRUE(analysis.isReachable(inLoop));
  EXPECT_TRUE(analysis.isReachable(afterLoop));

  // The loop's edge to itself does not keep it reachable once the label that
  // let control in has left.
  IRRewriter rewriter(&context, &analysis);
  Block *bb3 = &getFunc(*module).getBody().back();
  rewriter.moveOpBefore(&inLoop->getBlock()->front(), &bb3->front());
  EXPECT_FALSE(analysis.isReachable(inLoop));
  EXPECT_FALSE(analysis.isReachable(afterLoop));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, ErasedHolderBlock) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.return
    ^bb1:
      cir.label "in"
      cir.br ^bb2
    ^bb2:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_TRUE(analysis.isReachable(marker));

  // Erasing the label block takes away the only way into ^bb2.
  IRRewriter rewriter(&context, &analysis);
  rewriter.eraseBlock(&*std::next(getFunc(*module).getBody().begin()));
  EXPECT_FALSE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, InsertedBeforeHolder) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.goto "in"
    ^bb1:
      cir.scope {
        cir.br ^bb2
      ^bb2:
        cir.label "in"
        cir.yield
      }
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_TRUE(analysis.isReachable(marker));

  // Control enters ^bb1 at the scope, not at an operation put before it.
  IRRewriter rewriter(&context, &analysis);
  rewriter.setInsertionPointToStart(marker->getBlock());
  Operation *before = cir::ConstantOp::create(
      rewriter, marker->getLoc(), cast<cir::ConstantOp>(marker).getValue());
  EXPECT_FALSE(analysis.isReachable(before));
  EXPECT_TRUE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, CreatedHolder) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.return
    ^bb1:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_FALSE(analysis.isReachable(marker));

  // The label is built into the scope's region before the region belongs to
  // the scope, yet the scope holds it once inserted.
  IRRewriter rewriter(&context, &analysis);
  Operation *ret = marker->getBlock()->getTerminator();
  rewriter.setInsertionPoint(ret);
  cir::ScopeOp::create(rewriter, ret->getLoc(),
                       [](OpBuilder &builder, Location loc) {
                         cir::LabelOp::create(builder, loc, "in");
                         cir::YieldOp::create(builder, loc);
                       });
  EXPECT_FALSE(analysis.isReachable(marker));
  EXPECT_TRUE(analysis.isReachable(ret));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, NewEntryBlock) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    ^bb1:
      %1 = cir.const #cir.int<2> : !s32i
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *oldEntry = getMarker(*module, 1);
  Operation *marker = getMarker(*module, 2);
  EXPECT_FALSE(analysis.isReachable(marker));

  // A new entry block that branches to ^bb1 leaves the old entry block dead.
  IRRewriter rewriter(&context, &analysis);
  Region &body = getFunc(*module).getBody();
  rewriter.createBlock(&body, body.begin());
  cir::BrOp::create(rewriter, marker->getLoc(), marker->getBlock());
  EXPECT_TRUE(analysis.isReachable(marker));
  EXPECT_FALSE(analysis.isReachable(oldEntry));
  expectFresh(analysis, *module);
}

TEST_F(GotoAnalysisTest, BlockMovedAwayAndBack) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.br ^bb1
    ^bb1:
      cir.br ^bb2
    ^bb2:
      %0 = cir.const #cir.int<1> : !s32i
      cir.return
    }
    cir.func @g() {
      cir.return
    }
  )CIR");
  ASSERT_TRUE(module);
  cir::GotoAnalysis analysis(*module);
  Operation *marker = getMarker(*module, 1);
  EXPECT_TRUE(analysis.isReachable(marker));

  // The block comes back to the same address it left from.
  IRRewriter rewriter(&context, &analysis);
  Block *bb2 = marker->getBlock();
  Region &fBody = getFunc(*module, 0).getBody();
  Region &gBody = getFunc(*module, 1).getBody();
  rewriter.moveBlockBefore(bb2, &gBody, gBody.end());
  rewriter.moveBlockBefore(bb2, &fBody, fBody.end());
  EXPECT_TRUE(analysis.isReachable(marker));
  expectFresh(analysis, *module);
}

} // namespace
