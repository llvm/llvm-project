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

// Test that we can put a value into an int-type variable and load it
// back from that variable. Test what happens if default bindings are used.
class VariableBindConsumer : public ExprEngineConsumer {
  void performTest(const Decl *D) {
    StoreManager &StMgr = Eng.getStoreManager();
    SValBuilder &SVB = Eng.getSValBuilder();
    MemRegionManager &MRMgr = StMgr.getRegionManager();
    const ASTContext &ACtx = Eng.getContext();

    const auto *VDX0 = findDeclByName<VarDecl>(D, "x0");
    const auto *VDY0 = findDeclByName<VarDecl>(D, "y0");
    const auto *VDZ0 = findDeclByName<VarDecl>(D, "z0");
    const auto *VDX1 = findDeclByName<VarDecl>(D, "x1");
    const auto *VDY1 = findDeclByName<VarDecl>(D, "y1");
    assert(VDX0 && VDY0 && VDZ0 && VDX1 && VDY1);

    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);

    Loc LX0 = loc::MemRegionVal(MRMgr.getVarRegion(VDX0, SFC));
    Loc LY0 = loc::MemRegionVal(MRMgr.getVarRegion(VDY0, SFC));
    Loc LZ0 = loc::MemRegionVal(MRMgr.getVarRegion(VDZ0, SFC));
    Loc LX1 = loc::MemRegionVal(MRMgr.getVarRegion(VDX1, SFC));
    Loc LY1 = loc::MemRegionVal(MRMgr.getVarRegion(VDY1, SFC));

    Store StInit = StMgr.getInitialStore(SFC).getStore();
    SVal Zero = SVB.makeZeroVal(ACtx.IntTy);
    SVal One = SVB.makeIntVal(1, ACtx.IntTy);
    SVal NarrowZero = SVB.makeZeroVal(ACtx.CharTy);

    // Bind(Zero)
    Store StX0 =
        StMgr.Bind(StInit, LX0, Zero).getStore();
    ASSERT_EQ(Zero, StMgr.getBinding(StX0, LX0, ACtx.IntTy));

    // BindDefaultInitial(Zero)
    Store StY0 =
        StMgr.BindDefaultInitial(StInit, LY0.getAsRegion(), Zero).getStore();
    ASSERT_EQ(Zero, StMgr.getBinding(StY0, LY0, ACtx.IntTy));
    ASSERT_EQ(Zero, *StMgr.getDefaultBinding(StY0, LY0.getAsRegion()));

    // BindDefaultZero()
    Store StZ0 =
        StMgr.BindDefaultZero(StInit, LZ0.getAsRegion()).getStore();
    // BindDefaultZero wipes the region with '0 S8b', not with out Zero.
    // Direct load, however, does give us back the object of the type
    // that we specify for loading.
    ASSERT_EQ(Zero, StMgr.getBinding(StZ0, LZ0, ACtx.IntTy));
    ASSERT_EQ(NarrowZero, *StMgr.getDefaultBinding(StZ0, LZ0.getAsRegion()));

    // Bind(One)
    Store StX1 =
        StMgr.Bind(StInit, LX1, One).getStore();
    ASSERT_EQ(One, StMgr.getBinding(StX1, LX1, ACtx.IntTy));

    // BindDefaultInitial(One)
    Store StY1 =
        StMgr.BindDefaultInitial(StInit, LY1.getAsRegion(), One).getStore();
    ASSERT_EQ(One, StMgr.getBinding(StY1, LY1, ACtx.IntTy));
    ASSERT_EQ(One, *StMgr.getDefaultBinding(StY1, LY1.getAsRegion()));
  }

public:
  VariableBindConsumer(CompilerInstance &C) : ExprEngineConsumer(C) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const auto *D : DG)
      performTest(D);
    return true;
  }
};

class VariableBindAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    return llvm::make_unique<VariableBindConsumer>(Compiler);
  }
};

TEST(Store, VariableBind) {
  EXPECT_TRUE(tooling::runToolOnCode(
      new VariableBindAction, "void foo() { int x0, y0, z0, x1, y1; }"));
}

} // namespace
} // namespace ento
} // namespace clang
