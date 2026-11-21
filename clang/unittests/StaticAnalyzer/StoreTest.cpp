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

    const StackFrame *SF =
        Eng.getAnalysisDeclContextManager().getTopStackFrame(D);

    Loc LX0 = loc::MemRegionVal(MRManager.getVarRegion(VDX0, SF));
    Loc LY0 = loc::MemRegionVal(MRManager.getVarRegion(VDY0, SF));
    Loc LZ0 = loc::MemRegionVal(MRManager.getVarRegion(VDZ0, SF));
    Loc LX1 = loc::MemRegionVal(MRManager.getVarRegion(VDX1, SF));
    Loc LY1 = loc::MemRegionVal(MRManager.getVarRegion(VDY1, SF));

    // Keep the StoreRefs alive, a raw Store does not keep its bindings alive.
    StoreRef StInit = SManager.getInitialStore(SF);
    SVal Zero = Builder.makeZeroVal(ASTCtxt.IntTy);
    SVal One = Builder.makeIntVal(1, ASTCtxt.IntTy);
    SVal NarrowZero = Builder.makeZeroVal(ASTCtxt.CharTy);

    // Bind(Zero)
    StoreRef StX0 = SManager.Bind(StInit.getStore(), LX0, Zero).ResultingStore;
    EXPECT_EQ(Zero, SManager.getBinding(StX0.getStore(), LX0, ASTCtxt.IntTy));

    // BindDefaultInitial(Zero)
    StoreRef StY0 =
        SManager.BindDefaultInitial(StInit.getStore(), LY0.getAsRegion(), Zero)
            .ResultingStore;
    EXPECT_EQ(Zero, SManager.getBinding(StY0.getStore(), LY0, ASTCtxt.IntTy));
    EXPECT_EQ(Zero,
              *SManager.getDefaultBinding(StY0.getStore(), LY0.getAsRegion()));

    // BindDefaultZero()
    StoreRef StZ0 =
        SManager.BindDefaultZero(StInit.getStore(), LZ0.getAsRegion())
            .ResultingStore;
    // BindDefaultZero wipes the region with '0 S8b', not with out Zero.
    // Direct load, however, does give us back the object of the type
    // that we specify for loading.
    EXPECT_EQ(Zero, SManager.getBinding(StZ0.getStore(), LZ0, ASTCtxt.IntTy));
    EXPECT_EQ(NarrowZero,
              *SManager.getDefaultBinding(StZ0.getStore(), LZ0.getAsRegion()));

    // Bind(One)
    StoreRef StX1 = SManager.Bind(StInit.getStore(), LX1, One).ResultingStore;
    EXPECT_EQ(One, SManager.getBinding(StX1.getStore(), LX1, ASTCtxt.IntTy));

    // BindDefaultInitial(One)
    StoreRef StY1 =
        SManager.BindDefaultInitial(StInit.getStore(), LY1.getAsRegion(), One)
            .ResultingStore;
    EXPECT_EQ(One, SManager.getBinding(StY1.getStore(), LY1, ASTCtxt.IntTy));
    EXPECT_EQ(One,
              *SManager.getDefaultBinding(StY1.getStore(), LY1.getAsRegion()));
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

    const StackFrame *SF =
        Eng.getAnalysisDeclContextManager().getTopStackFrame(D);

    QualType Int = ASTCtxt.IntTy;

    // Get region for 'test'
    const SubRegion *CLRegion = MRManager.getCompoundLiteralRegion(CL, SF);

    // Get value for 'test[0]'
    NonLoc Zero = Builder.makeIntVal(0, false);
    loc::MemRegionVal ZeroElement(
        MRManager.getElementRegion(ASTCtxt.IntTy, Zero, CLRegion, ASTCtxt));

    StoreRef StInit = SManager.getInitialStore(SF);
    // Let's bind constant 1 to 'test[0]'
    SVal One = Builder.makeIntVal(1, Int);
    StoreRef StX =
        SManager.Bind(StInit.getStore(), ZeroElement, One).ResultingStore;

    // And make sure that we can read this binding back as it was
    EXPECT_EQ(One, SManager.getBinding(StX.getStore(), ZeroElement, Int));
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
