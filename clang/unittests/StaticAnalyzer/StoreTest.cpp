//===- unittests/StaticAnalyzer/StoreTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Reusables.h"

#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

class StoreTestConsumer : public ExprEngineConsumer {
public:
  StoreTestConsumer(CompilerInstance &C) : ExprEngineConsumer(C) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const auto *D : DG)
      performTest(D);
    return true;
  }

private:
  virtual void performTest(const Decl *D) = 0;
};

template <class ConsumerTy> class TestAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    return std::make_unique<ConsumerTy>(Compiler);
  }
};

// Test that we can put a value into an int-type variable and load it
// back from that variable. Test what happens if default bindings are used.
class VariableBindConsumer : public StoreTestConsumer {
  void performTest(const Decl *D) override {
    StoreManager &SManager = Eng.getStoreManager();
    SValBuilder &Builder = Eng.getSValBuilder();
    MemRegionManager &MRManager = SManager.getRegionManager();
    const ASTContext &ASTCtxt = Eng.getContext();

    const auto *VDX0 = findDeclByName<VarDecl>(D, "x0");
    const auto *VDY0 = findDeclByName<VarDecl>(D, "y0");
    const auto *VDZ0 = findDeclByName<VarDecl>(D, "z0");
    const auto *VDX1 = findDeclByName<VarDecl>(D, "x1");
    const auto *VDY1 = findDeclByName<VarDecl>(D, "y1");

    ASSERT_TRUE(VDX0 && VDY0 && VDZ0 && VDX1 && VDY1);

    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);

    Loc LX0 = loc::MemRegionVal(MRManager.getVarRegion(VDX0, SFC));
    Loc LY0 = loc::MemRegionVal(MRManager.getVarRegion(VDY0, SFC));
    Loc LZ0 = loc::MemRegionVal(MRManager.getVarRegion(VDZ0, SFC));
    Loc LX1 = loc::MemRegionVal(MRManager.getVarRegion(VDX1, SFC));
    Loc LY1 = loc::MemRegionVal(MRManager.getVarRegion(VDY1, SFC));

    Store StInit = SManager.getInitialStore(SFC).getStore();
    SVal Zero = Builder.makeZeroVal(ASTCtxt.IntTy);
    SVal One = Builder.makeIntVal(1, ASTCtxt.IntTy);
    SVal NarrowZero = Builder.makeZeroVal(ASTCtxt.CharTy);

    // Bind(Zero)
    Store StX0 = SManager.Bind(StInit, LX0, Zero).getStore();
    EXPECT_EQ(Zero, SManager.getBinding(StX0, LX0, ASTCtxt.IntTy));

    // BindDefaultInitial(Zero)
    Store StY0 =
        SManager.BindDefaultInitial(StInit, LY0.getAsRegion(), Zero).getStore();
    EXPECT_EQ(Zero, SManager.getBinding(StY0, LY0, ASTCtxt.IntTy));
    EXPECT_EQ(Zero, *SManager.getDefaultBinding(StY0, LY0.getAsRegion()));

    // BindDefaultZero()
    Store StZ0 = SManager.BindDefaultZero(StInit, LZ0.getAsRegion()).getStore();
    // BindDefaultZero wipes the region with '0 S8b', not with out Zero.
    // Direct load, however, does give us back the object of the type
    // that we specify for loading.
    EXPECT_EQ(Zero, SManager.getBinding(StZ0, LZ0, ASTCtxt.IntTy));
    EXPECT_EQ(NarrowZero, *SManager.getDefaultBinding(StZ0, LZ0.getAsRegion()));

    // Bind(One)
    Store StX1 = SManager.Bind(StInit, LX1, One).getStore();
    EXPECT_EQ(One, SManager.getBinding(StX1, LX1, ASTCtxt.IntTy));

    // BindDefaultInitial(One)
    Store StY1 =
        SManager.BindDefaultInitial(StInit, LY1.getAsRegion(), One).getStore();
    EXPECT_EQ(One, SManager.getBinding(StY1, LY1, ASTCtxt.IntTy));
    EXPECT_EQ(One, *SManager.getDefaultBinding(StY1, LY1.getAsRegion()));
  }

public:
  using StoreTestConsumer::StoreTestConsumer;
};

TEST(Store, VariableBind) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::make_unique<TestAction<VariableBindConsumer>>(),
      "void foo() { int x0, y0, z0, x1, y1; }"));
}

class LiteralCompoundConsumer : public StoreTestConsumer {
  void performTest(const Decl *D) override {
    StoreManager &SManager = Eng.getStoreManager();
    SValBuilder &Builder = Eng.getSValBuilder();
    MemRegionManager &MRManager = SManager.getRegionManager();
    ASTContext &ASTCtxt = Eng.getContext();

    using namespace ast_matchers;

    const auto *CL = findNode<CompoundLiteralExpr>(D, compoundLiteralExpr());

    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);

    QualType Int = ASTCtxt.IntTy;

    // Get region for 'test'
    const SubRegion *CLRegion = MRManager.getCompoundLiteralRegion(CL, SFC);

    // Get value for 'test[0]'
    NonLoc Zero = Builder.makeIntVal(0, false);
    loc::MemRegionVal ZeroElement(
        MRManager.getElementRegion(ASTCtxt.IntTy, Zero, CLRegion, ASTCtxt));

    Store StInit = SManager.getInitialStore(SFC).getStore();
    // Let's bind constant 1 to 'test[0]'
    SVal One = Builder.makeIntVal(1, Int);
    Store StX = SManager.Bind(StInit, ZeroElement, One).getStore();

    // And make sure that we can read this binding back as it was
    EXPECT_EQ(One, SManager.getBinding(StX, ZeroElement, Int));
  }

public:
  using StoreTestConsumer::StoreTestConsumer;
};

TEST(Store, LiteralCompound) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::make_unique<TestAction<LiteralCompoundConsumer>>(),
      "void foo() { int *test = (int[]){ 1, 2, 3 }; }", "input.c"));
}

} // namespace
} // namespace ento
} // namespace clang
