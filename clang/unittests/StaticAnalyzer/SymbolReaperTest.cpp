//===- unittests/StaticAnalyzer/SymbolReaperTest.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

using namespace ast_matchers;

// A re-usable consumer that constructs ExprEngine out of CompilerInvocation.
// TODO: Actually re-use it when we write our second test.
class ExprEngineConsumer : public ASTConsumer {
protected:
  CompilerInstance &C;

private:
  // We need to construct all of these in order to construct ExprEngine.
  CheckerManager ChkMgr;
  cross_tu::CrossTranslationUnitContext CTU;
  PathDiagnosticConsumers Consumers;
  AnalysisManager AMgr;
  SetOfConstDecls VisitedCallees;
  FunctionSummariesTy FS;

protected:
  ExprEngine Eng;

  // Find a declaration in the current AST by name. This has nothing to do
  // with ExprEngine but turns out to be handy.
  // TODO: There's probably a better place for it.
  template <typename T>
  const T *findDeclByName(const Decl *Where, StringRef Name) {
    auto Matcher = decl(hasDescendant(namedDecl(hasName(Name)).bind("d")));
    auto Matches = match(Matcher, *Where, Eng.getContext());
    assert(Matches.size() == 1 && "Ambiguous name!");
    const T *Node = selectFirst<T>("d", Matches);
    assert(Node && "Name not found!");
    return Node;
  }

public:
  ExprEngineConsumer(CompilerInstance &C)
      : C(C), ChkMgr(C.getASTContext(), *C.getAnalyzerOpts()), CTU(C),
        Consumers(),
        AMgr(C.getASTContext(), C.getDiagnostics(), Consumers,
             CreateRegionStoreManager, CreateRangeConstraintManager, &ChkMgr,
             *C.getAnalyzerOpts()),
        VisitedCallees(), FS(),
        Eng(CTU, AMgr, &VisitedCallees, &FS, ExprEngine::Inline_Regular) {}
};

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

class SuperRegionLivenessAction: public ASTFrontendAction {
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
