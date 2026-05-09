//===- ControlFlowTest.cpp - Unit tests for CIR control flow interfaces ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <gtest/gtest.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Test helpers
//===----------------------------------------------------------------------===//

/// Use nullptr in `expected` to represent the parent (exit).
static void expectRegionSuccessors(ArrayRef<RegionSuccessor> actual,
                                   ArrayRef<Region *> expected,
                                   StringRef label) {
  EXPECT_EQ(actual.size(), expected.size())
      << "successor count mismatch from " << label.str();
  for (Region *r : expected)
    EXPECT_TRUE(llvm::any_of(
        actual,
        [r](const RegionSuccessor &s) { return s.getSuccessor() == r; }))
        << "expected region "
        << (r ? std::to_string(r->getRegionNumber()) : "parent")
        << " not found in successors from " << label.str();
  for (const RegionSuccessor &s : actual) {
    Region *r = s.getSuccessor();
    EXPECT_TRUE(llvm::is_contained(expected, r))
        << "unexpected region "
        << (r ? std::to_string(r->getRegionNumber()) : "parent")
        << " in successors from " << label.str();
  }
}

/// Check that `op.getSuccessorRegions(point)` matches `expected`.
static void expectSuccessors(RegionBranchOpInterface op,
                             RegionBranchPoint point,
                             ArrayRef<Region *> expected) {
  SmallVector<RegionSuccessor> successors;
  op.getSuccessorRegions(point, successors);
  expectRegionSuccessors(successors, expected,
                         point.isParent() ? "parent" : "region terminator");
}

/// Return the RegionBranchTerminatorOpInterface for the region's terminator,
/// failing the test if the terminator doesn't implement the interface.
static RegionBranchTerminatorOpInterface getTerminator(Region &region) {
  Operation *term = region.front().getTerminator();
  auto branchTerm = dyn_cast<RegionBranchTerminatorOpInterface>(term);
  EXPECT_TRUE(branchTerm)
      << "terminator '" << term->getName().getStringRef().str()
      << "' in region #" << region.getRegionNumber()
      << " does not implement RegionBranchTerminatorOpInterface";
  return branchTerm;
}

/// Check that the terminator of `region` has the given successor regions.
static void expectTerminatorSuccessors(Region &region,
                                       ArrayRef<Region *> expected) {
  RegionBranchTerminatorOpInterface term = getTerminator(region);
  if (!term)
    return;
  SmallVector<RegionSuccessor> successors;
  term.getSuccessorRegions(/*operands=*/{}, successors);
  expectRegionSuccessors(successors, expected,
                         "region #" + std::to_string(region.getRegionNumber()));
}

/// Verify control flow interface consistency beyond what mlir::verify checks:
///   - Every non-empty region is reachable
///   - Every terminator implements RegionBranchTerminatorOpInterface
///   - Op and terminator agree on successor regions
static void verifyControlFlowInterfaceConsistency(RegionBranchOpInterface op) {
  EXPECT_TRUE(succeeded(mlir::verify(op)))
      << "MLIR verifier failed for '" << op->getName().getStringRef().str()
      << "'";

  SmallVector<RegionSuccessor> entrySuccessors;
  op.getSuccessorRegions(RegionBranchPoint::parent(), entrySuccessors);
  llvm::SmallPtrSet<Region *, 4> allReachable;
  for (auto &succ : entrySuccessors)
    allReachable.insert(succ.getSuccessor());
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    RegionBranchTerminatorOpInterface term = getTerminator(region);
    SmallVector<RegionSuccessor> opSuccessors;
    op.getSuccessorRegions(region, opSuccessors);
    for (auto &succ : opSuccessors)
      allReachable.insert(succ.getSuccessor());

    // Op and terminator should report the same successors.
    if (term) {
      SmallVector<RegionSuccessor> termSuccessors;
      term.getSuccessorRegions(/*operands=*/{}, termSuccessors);
      SmallPtrSet<Region *, 4> opSet, termSet;
      for (auto &s : opSuccessors)
        opSet.insert(s.getSuccessor());
      for (auto &s : termSuccessors)
        termSet.insert(s.getSuccessor());
      EXPECT_EQ(opSet, termSet)
          << "op and terminator disagree on successors from region #"
          << region.getRegionNumber();
    }
  }
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    EXPECT_TRUE(allReachable.contains(&region))
        << "Region #" << region.getRegionNumber()
        << " is non-empty but not reachable from any branch point";
  }
  EXPECT_TRUE(allReachable.contains(nullptr))
      << "parent (exit) not reachable from any branch point";
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class CIRControlFlowTest : public ::testing::Test {
protected:
  CIRControlFlowTest() { context.loadDialect<cir::CIRDialect>(); }

  OwningOpRef<ModuleOp> parse(StringRef ir) {
    auto module = parseSourceString<ModuleOp>(ir, &context);
    EXPECT_TRUE(module) << "failed to parse IR";
    return module;
  }

  template <typename T> T findFirstOp(ModuleOp module) {
    T result = nullptr;
    module->walk([&](T op) {
      result = op;
      return WalkResult::interrupt();
    });
    EXPECT_NE(result, nullptr) << "op not found in module";
    return result;
  }

  static RegionBranchOpInterface asRegionBranch(Operation *op) {
    return cast<RegionBranchOpInterface>(op);
  }

  MLIRContext context;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(CIRControlFlowTest, IfOpThenOnly) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%cond : !cir.bool) {
      cir.if %cond {
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto ifOp = findFirstOp<cir::IfOp>(*module);

  // Parent branches to then or exits (no else).
  expectSuccessors(ifOp, RegionBranchPoint::parent(),
                   {&ifOp.getThenRegion(), nullptr});
  expectTerminatorSuccessors(ifOp.getThenRegion(), {nullptr});

  RegionBranchOpInterface ifBranch = asRegionBranch(ifOp);
  EXPECT_FALSE(ifBranch.isRepetitiveRegion(0));
  EXPECT_FALSE(ifBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(ifOp);
}

TEST_F(CIRControlFlowTest, IfOpThenElse) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%cond : !cir.bool) {
      cir.if %cond {
        cir.yield
      } else {
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto ifOp = findFirstOp<cir::IfOp>(*module);

  expectSuccessors(ifOp, RegionBranchPoint::parent(),
                   {&ifOp.getThenRegion(), &ifOp.getElseRegion()});
  expectTerminatorSuccessors(ifOp.getThenRegion(), {nullptr});
  expectTerminatorSuccessors(ifOp.getElseRegion(), {nullptr});

  RegionBranchOpInterface ifBranch = asRegionBranch(ifOp);
  EXPECT_FALSE(ifBranch.isRepetitiveRegion(0));
  EXPECT_FALSE(ifBranch.isRepetitiveRegion(1));
  EXPECT_FALSE(ifBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(ifOp);
}

TEST_F(CIRControlFlowTest, ScopeOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f() {
      cir.scope {
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto scopeOp = findFirstOp<cir::ScopeOp>(*module);

  expectSuccessors(scopeOp, RegionBranchPoint::parent(),
                   {&scopeOp.getScopeRegion()});
  expectTerminatorSuccessors(scopeOp.getScopeRegion(), {nullptr});

  RegionBranchOpInterface scopeBranch = asRegionBranch(scopeOp);
  EXPECT_FALSE(scopeBranch.isRepetitiveRegion(0));
  EXPECT_FALSE(scopeBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(scopeOp);
}

TEST_F(CIRControlFlowTest, ScopeOpWithResult) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    !s32i = !cir.int<s, 32>
    cir.func @f() -> !s32i {
      %0 = cir.scope {
        %c = cir.const #cir.int<42> : !s32i
        cir.yield %c : !s32i
      } : !s32i
      cir.return %0 : !s32i
    }
  )CIR");
  auto scopeOp = findFirstOp<cir::ScopeOp>(*module);

  // getSuccessorInputs(parent) should return the scope's result.
  ValueRange parentInputs =
      scopeOp.getSuccessorInputs(RegionSuccessor::parent());
  EXPECT_EQ(parentInputs.size(), 1u);

  // The yield's operands are forwarded to the parent result.
  RegionBranchTerminatorOpInterface term =
      getTerminator(scopeOp.getScopeRegion());
  ASSERT_TRUE(term);
  OperandRange yieldOperands =
      term.getSuccessorOperands(RegionSuccessor::parent());
  EXPECT_EQ(yieldOperands.size(), 1u);

  verifyControlFlowInterfaceConsistency(scopeOp);
}

TEST_F(CIRControlFlowTest, TernaryOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    !u32i = !cir.int<u, 32>
    cir.func @f(%cond : !cir.bool) -> !u32i {
      %0 = cir.ternary(%cond, true {
        %a = cir.const #cir.int<0> : !u32i
        cir.yield %a : !u32i
      }, false {
        %b = cir.const #cir.int<1> : !u32i
        cir.yield %b : !u32i
      }) : (!cir.bool) -> !u32i
      cir.return %0 : !u32i
    }
  )CIR");
  auto ternOp = findFirstOp<cir::TernaryOp>(*module);

  expectSuccessors(ternOp, RegionBranchPoint::parent(),
                   {&ternOp.getTrueRegion(), &ternOp.getFalseRegion()});
  expectTerminatorSuccessors(ternOp.getTrueRegion(), {nullptr});
  expectTerminatorSuccessors(ternOp.getFalseRegion(), {nullptr});

  RegionBranchOpInterface ternBranch = asRegionBranch(ternOp);
  EXPECT_FALSE(ternBranch.isRepetitiveRegion(0));
  EXPECT_FALSE(ternBranch.isRepetitiveRegion(1));
  EXPECT_FALSE(ternBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(ternOp);
}

TEST_F(CIRControlFlowTest, SwitchOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    !s32i = !cir.int<s, 32>
    cir.func @f(%val : !s32i) {
      cir.switch (%val : !s32i) {
        cir.case (equal, [#cir.int<1> : !s32i]) {
          cir.yield
        }
        cir.case (default, []) {
          cir.yield
        }
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto switchOp = findFirstOp<cir::SwitchOp>(*module);

  expectSuccessors(switchOp, RegionBranchPoint::parent(),
                   {&switchOp.getBody()});
  expectTerminatorSuccessors(switchOp.getBody(), {nullptr});

  RegionBranchOpInterface switchBranch = asRegionBranch(switchOp);
  EXPECT_FALSE(switchBranch.isRepetitiveRegion(0));
  EXPECT_FALSE(switchBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(switchOp);
}

TEST_F(CIRControlFlowTest, WhileOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%cond : !cir.bool) {
      cir.while {
        cir.condition(%cond)
      } do {
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto whileOp = findFirstOp<cir::WhileOp>(*module);

  // Parent enters the condition region.
  expectSuccessors(whileOp, RegionBranchPoint::parent(), {&whileOp.getCond()});

  // Condition branches to body or exits.
  expectTerminatorSuccessors(whileOp.getCond(), {&whileOp.getBody(), nullptr});

  // Body branches back to condition (loop back-edge).
  expectTerminatorSuccessors(whileOp.getBody(), {&whileOp.getCond()});

  RegionBranchOpInterface whileBranch = asRegionBranch(whileOp);
  EXPECT_TRUE(whileBranch.isRepetitiveRegion(0)); // cond
  EXPECT_TRUE(whileBranch.isRepetitiveRegion(1)); // body
  EXPECT_TRUE(whileBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(whileOp);
}

TEST_F(CIRControlFlowTest, ForOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%cond : !cir.bool) {
      cir.for : cond {
        cir.condition(%cond)
      } body {
        cir.yield
      } step {
        cir.yield
      }
      cir.return
    }
  )CIR");
  auto forOp = findFirstOp<cir::ForOp>(*module);

  // Parent enters the condition region.
  expectSuccessors(forOp, RegionBranchPoint::parent(), {&forOp.getCond()});

  // Condition branches to body or exits.
  expectTerminatorSuccessors(forOp.getCond(), {&forOp.getBody(), nullptr});

  // Body goes to step.
  expectTerminatorSuccessors(forOp.getBody(), {&forOp.getStep()});

  // Step goes back to condition.
  expectTerminatorSuccessors(forOp.getStep(), {&forOp.getCond()});

  RegionBranchOpInterface forBranch = asRegionBranch(forOp);
  EXPECT_TRUE(forBranch.isRepetitiveRegion(0)); // cond
  EXPECT_TRUE(forBranch.isRepetitiveRegion(1)); // body
  EXPECT_TRUE(forBranch.isRepetitiveRegion(2)); // step
  EXPECT_TRUE(forBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(forOp);
}

TEST_F(CIRControlFlowTest, DoWhileOp) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    cir.func @f(%cond : !cir.bool) {
      cir.do {
        cir.yield
      } while {
        cir.condition(%cond)
      }
      cir.return
    }
  )CIR");
  auto doWhileOp = findFirstOp<cir::DoWhileOp>(*module);

  // Parent enters the body region (not condition).
  expectSuccessors(doWhileOp, RegionBranchPoint::parent(),
                   {&doWhileOp.getBody()});

  // Body goes to condition.
  expectTerminatorSuccessors(doWhileOp.getBody(), {&doWhileOp.getCond()});

  // Condition branches back to body or exits.
  expectTerminatorSuccessors(doWhileOp.getCond(),
                             {&doWhileOp.getBody(), nullptr});

  RegionBranchOpInterface doWhileBranch = asRegionBranch(doWhileOp);
  EXPECT_TRUE(doWhileBranch.isRepetitiveRegion(0)); // body
  EXPECT_TRUE(doWhileBranch.isRepetitiveRegion(1)); // cond
  EXPECT_TRUE(doWhileBranch.hasLoop());

  verifyControlFlowInterfaceConsistency(doWhileOp);
}

TEST_F(CIRControlFlowTest, TryOpWithCatchAll) {
  OwningOpRef<ModuleOp> module = parse(R"CIR(
    !void = !cir.void
    cir.func @f() {
      cir.scope {
        cir.try {
          cir.yield
        } catch all (%eh : !cir.eh_token) {
          %ct, %exn = cir.begin_catch %eh
            : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
          cir.cleanup.scope {
            cir.yield
          } cleanup eh {
            cir.end_catch %ct : !cir.catch_token
            cir.yield
          }
          cir.yield
        }
      }
      cir.return
    }
  )CIR");
  auto tryOp = findFirstOp<cir::TryOp>(*module);

  Region &tryRegion = tryOp.getTryRegion();
  MutableArrayRef<Region> handlerRegions = tryOp.getHandlerRegions();
  ASSERT_EQ(handlerRegions.size(), 1u);

  expectSuccessors(tryOp, RegionBranchPoint::parent(),
                   {&tryRegion, &handlerRegions[0]});
  expectTerminatorSuccessors(tryRegion, {nullptr});
  expectTerminatorSuccessors(handlerRegions[0], {nullptr});

  EXPECT_FALSE(asRegionBranch(tryOp).hasLoop());

  // TODO: TryOp::getSuccessorInputs returns empty for handler regions that
  // have block arguments, so verifyControlFlowInterfaceConsistency fails.
}
