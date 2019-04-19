//===- unittests/StaticAnalyzer/Reusables.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_STATICANALYZER_REUSABLES_H
#define LLVM_CLANG_UNITTESTS_STATICANALYZER_REUSABLES_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

namespace clang {
namespace ento {

// Find a declaration in the current AST by name.
template <typename T>
const T *findDeclByName(const Decl *Where, StringRef Name) {
  using namespace ast_matchers;
  auto Matcher = decl(hasDescendant(namedDecl(hasName(Name)).bind("d")));
  auto Matches = match(Matcher, *Where, Where->getASTContext());
  assert(Matches.size() == 1 && "Ambiguous name!");
  const T *Node = selectFirst<T>("d", Matches);
  assert(Node && "Name not found!");
  return Node;
}

// A re-usable consumer that constructs ExprEngine out of CompilerInvocation.
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

} // namespace ento
} // namespace clang

#endif
