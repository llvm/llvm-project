//===- unittests/StaticAnalyzer/SymbolReaperTest.cpp ----------------------===//
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

class SuperRegionLivenessConsumer : public ExprEngineConsumer {
  void performTest(const Decl *D) {
    const auto *FD = findDeclByName<FieldDecl>(D, "x");
    const auto *VD = findDeclByName<VarDecl>(D, "s");
    assert(FD && VD);

    // The variable must belong to a stack frame,
    // otherwise SymbolReaper would think it's a global.
    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);

    // Create regions for 's' and 's.x'.
    const VarRegion *VR = Eng.getRegionManager().getVarRegion(VD, SFC);
    const FieldRegion *FR = Eng.getRegionManager().getFieldRegion(FD, VR);

    // Pass a null location context to the SymbolReaper so that
    // it was thinking that the variable is dead.
    SymbolReaper SymReaper((StackFrameContext *)nullptr, (Stmt *)nullptr,
                           Eng.getSymbolManager(), Eng.getStoreManager());

    SymReaper.markLive(FR);
    EXPECT_TRUE(SymReaper.isLiveRegion(VR));
  }

public:
  SuperRegionLivenessConsumer(CompilerInstance &C) : ExprEngineConsumer(C) {}
  ~SuperRegionLivenessConsumer() override {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const auto *D : DG)
      performTest(D);
    return true;
  }
};

class SuperRegionLivenessAction : public ASTFrontendAction {
public:
  SuperRegionLivenessAction() {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    return llvm::make_unique<SuperRegionLivenessConsumer>(Compiler);
  }
};

// Test that marking s.x as live would also make s live.
TEST(SymbolReaper, SuperRegionLiveness) {
  EXPECT_TRUE(tooling::runToolOnCode(new SuperRegionLivenessAction,
                                     "void foo() { struct S { int x; } s; }"));
}

} // namespace
} // namespace ento
} // namespace clang
