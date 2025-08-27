//===- unittests/Analysis/FlowSensitive/CachedConstAccessorsLatticeTest.cpp ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/CachedConstAccessorsLattice.h"

#include <cassert>
#include <memory>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "clang/Testing/TestAST.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::dataflow {
namespace {

using ast_matchers::BoundNodes;
using ast_matchers::callee;
using ast_matchers::cxxMemberCallExpr;
using ast_matchers::functionDecl;
using ast_matchers::hasName;
using ast_matchers::match;
using ast_matchers::selectFirst;

using dataflow::DataflowAnalysisContext;
using dataflow::Environment;
using dataflow::LatticeJoinEffect;
using dataflow::RecordStorageLocation;
using dataflow::Value;
using dataflow::WatchedLiteralsSolver;

using testing::SizeIs;

NamedDecl *lookup(StringRef Name, const DeclContext &DC) {
  auto Result = DC.lookup(&DC.getParentASTContext().Idents.get(Name));
  EXPECT_TRUE(Result.isSingleResult()) << Name;
  return Result.front();
}

class CachedConstAccessorsLatticeTest : public ::testing::Test {
protected:
  using LatticeT = CachedConstAccessorsLattice<NoopLattice>;

  DataflowAnalysisContext DACtx{std::make_unique<WatchedLiteralsSolver>()};
  Environment Env{DACtx};
};

// Basic test AST with two const methods (return a value, and return a ref).
struct CommonTestInputs {
  CommonTestInputs()
      : AST(R"cpp(
    struct S {
      int *valProperty() const;
      int &refProperty() const;
    };
    void target() {
      S s;
      s.valProperty();
      S s2;
      s2.refProperty();
    }
  )cpp") {
    auto *SDecl = cast<CXXRecordDecl>(
        lookup("S", *AST.context().getTranslationUnitDecl()));
    SType = AST.context().getCanonicalTagType(SDecl);
    CallVal = selectFirst<CallExpr>(
        "call",
        match(cxxMemberCallExpr(callee(functionDecl(hasName("valProperty"))))
                  .bind("call"),
              AST.context()));
    assert(CallVal != nullptr);

    CallRef = selectFirst<CallExpr>(
        "call",
        match(cxxMemberCallExpr(callee(functionDecl(hasName("refProperty"))))
                  .bind("call"),
              AST.context()));
    assert(CallRef != nullptr);
  }

  TestAST AST;
  QualType SType;
  const CallExpr *CallVal;
  const CallExpr *CallRef;
};

TEST_F(CachedConstAccessorsLatticeTest,
       SamePrimitiveValBeforeClearOrDiffAfterClear) {
  CommonTestInputs Inputs;
  auto *CE = Inputs.CallVal;
  RecordStorageLocation Loc(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                            {});

  LatticeT Lattice;
  Value *Val1 = Lattice.getOrCreateConstMethodReturnValue(Loc, CE, Env);
  Value *Val2 = Lattice.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  EXPECT_EQ(Val1, Val2);

  Lattice.clearConstMethodReturnValues(Loc);
  Value *Val3 = Lattice.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  EXPECT_NE(Val3, Val1);
  EXPECT_NE(Val3, Val2);
}

TEST_F(CachedConstAccessorsLatticeTest, SameLocBeforeClearOrDiffAfterClear) {
  CommonTestInputs Inputs;
  auto *CE = Inputs.CallRef;
  RecordStorageLocation Loc(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                            {});

  LatticeT Lattice;
  auto NopInit = [](StorageLocation &) {};
  const FunctionDecl *Callee = CE->getDirectCallee();
  ASSERT_NE(Callee, nullptr);
  StorageLocation &Loc1 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NopInit);
  auto NotCalled = [](StorageLocation &) {
    ASSERT_TRUE(false) << "Not reached";
  };
  StorageLocation &Loc2 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NotCalled);

  EXPECT_EQ(&Loc1, &Loc2);

  Lattice.clearConstMethodReturnStorageLocations(Loc);
  StorageLocation &Loc3 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NopInit);

  EXPECT_NE(&Loc3, &Loc1);
  EXPECT_NE(&Loc3, &Loc2);
}

TEST_F(CachedConstAccessorsLatticeTest,
       SameStructValBeforeClearOrDiffAfterClear) {
  TestAST AST(R"cpp(
    struct S {
      S structValProperty() const;
    };
    void target() {
      S s;
      s.structValProperty();
    }
  )cpp");
  auto *SDecl =
      cast<CXXRecordDecl>(lookup("S", *AST.context().getTranslationUnitDecl()));
  CanQualType SType = AST.context().getCanonicalTagType(SDecl);
  const CallExpr *CE = selectFirst<CallExpr>(
      "call", match(cxxMemberCallExpr(
                        callee(functionDecl(hasName("structValProperty"))))
                        .bind("call"),
                    AST.context()));
  ASSERT_NE(CE, nullptr);

  RecordStorageLocation Loc(SType, RecordStorageLocation::FieldToLoc(), {});

  LatticeT Lattice;
  // Accessors that return a record by value are modeled by a record storage
  // location (instead of a Value).
  auto NopInit = [](StorageLocation &) {};
  const FunctionDecl *Callee = CE->getDirectCallee();
  ASSERT_NE(Callee, nullptr);
  StorageLocation &Loc1 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NopInit);
  auto NotCalled = [](StorageLocation &) {
    ASSERT_TRUE(false) << "Not reached";
  };
  StorageLocation &Loc2 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NotCalled);

  EXPECT_EQ(&Loc1, &Loc2);

  Lattice.clearConstMethodReturnStorageLocations(Loc);
  StorageLocation &Loc3 = Lattice.getOrCreateConstMethodReturnStorageLocation(
      Loc, Callee, Env, NopInit);

  EXPECT_NE(&Loc3, &Loc1);
  EXPECT_NE(&Loc3, &Loc1);
}

TEST_F(CachedConstAccessorsLatticeTest, ClearDifferentLocs) {
  CommonTestInputs Inputs;
  auto *CE = Inputs.CallRef;
  RecordStorageLocation LocS1(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                              {});
  RecordStorageLocation LocS2(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                              {});

  LatticeT Lattice;
  auto NopInit = [](StorageLocation &) {};
  const FunctionDecl *Callee = CE->getDirectCallee();
  ASSERT_NE(Callee, nullptr);
  StorageLocation &RetLoc1 =
      Lattice.getOrCreateConstMethodReturnStorageLocation(LocS1, Callee, Env,
                                                          NopInit);
  Lattice.clearConstMethodReturnStorageLocations(LocS2);
  auto NotCalled = [](StorageLocation &) {
    ASSERT_TRUE(false) << "Not reached";
  };
  StorageLocation &RetLoc2 =
      Lattice.getOrCreateConstMethodReturnStorageLocation(LocS1, Callee, Env,
                                                          NotCalled);

  EXPECT_EQ(&RetLoc1, &RetLoc2);
}

TEST_F(CachedConstAccessorsLatticeTest, DifferentValsFromDifferentLocs) {
  TestAST AST(R"cpp(
    struct S {
      int *valProperty() const;
    };
    void target() {
      S s1;
      s1.valProperty();
      S s2;
      s2.valProperty();
    }
  )cpp");
  auto *SDecl =
      cast<CXXRecordDecl>(lookup("S", *AST.context().getTranslationUnitDecl()));
  CanQualType SType = AST.context().getCanonicalTagType(SDecl);
  SmallVector<BoundNodes, 1> valPropertyCalls =
      match(cxxMemberCallExpr(callee(functionDecl(hasName("valProperty"))))
                .bind("call"),
            AST.context());
  ASSERT_THAT(valPropertyCalls, SizeIs(2));

  const CallExpr *CE1 = selectFirst<CallExpr>("call", valPropertyCalls);
  ASSERT_NE(CE1, nullptr);

  valPropertyCalls.erase(valPropertyCalls.begin());
  const CallExpr *CE2 = selectFirst<CallExpr>("call", valPropertyCalls);
  ASSERT_NE(CE2, nullptr);
  ASSERT_NE(CE1, CE2);

  RecordStorageLocation LocS1(SType, RecordStorageLocation::FieldToLoc(), {});
  RecordStorageLocation LocS2(SType, RecordStorageLocation::FieldToLoc(), {});

  LatticeT Lattice;
  Value *Val1 = Lattice.getOrCreateConstMethodReturnValue(LocS1, CE1, Env);
  Value *Val2 = Lattice.getOrCreateConstMethodReturnValue(LocS2, CE2, Env);

  EXPECT_NE(Val1, Val2);
}

TEST_F(CachedConstAccessorsLatticeTest, JoinSameNoop) {
  CommonTestInputs Inputs;
  auto *CE = Inputs.CallVal;
  RecordStorageLocation Loc(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                            {});

  LatticeT EmptyLattice;
  LatticeT EmptyLattice2;
  EXPECT_EQ(EmptyLattice.join(EmptyLattice2), LatticeJoinEffect::Unchanged);

  LatticeT Lattice1;
  Lattice1.getOrCreateConstMethodReturnValue(Loc, CE, Env);
  EXPECT_EQ(Lattice1.join(Lattice1), LatticeJoinEffect::Unchanged);
}

TEST_F(CachedConstAccessorsLatticeTest, ProducesNewValueAfterJoinDistinct) {
  CommonTestInputs Inputs;
  auto *CE = Inputs.CallVal;
  RecordStorageLocation Loc(Inputs.SType, RecordStorageLocation::FieldToLoc(),
                            {});

  // L1 w/ v vs L2 empty
  LatticeT Lattice1;
  Value *Val1 = Lattice1.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  LatticeT EmptyLattice;

  EXPECT_EQ(Lattice1.join(EmptyLattice), LatticeJoinEffect::Changed);
  Value *ValAfterJoin =
      Lattice1.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  EXPECT_NE(ValAfterJoin, Val1);

  // L1 w/ v1 vs L3 w/ v2
  LatticeT Lattice3;
  Value *Val3 = Lattice3.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  EXPECT_EQ(Lattice1.join(Lattice3), LatticeJoinEffect::Changed);
  Value *ValAfterJoin2 =
      Lattice1.getOrCreateConstMethodReturnValue(Loc, CE, Env);

  EXPECT_NE(ValAfterJoin2, ValAfterJoin);
  EXPECT_NE(ValAfterJoin2, Val3);
}

} // namespace
} // namespace clang::dataflow
