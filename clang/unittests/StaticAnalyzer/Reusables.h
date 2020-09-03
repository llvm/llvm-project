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

// Find a node in the current AST that matches a matcher.
template <typename T, typename MatcherT>
const T *findNode(const Decl *Where, MatcherT What) {
  using namespace ast_matchers;
  auto Matches = match(decl(hasDescendant(What.bind("root"))),
                       *Where, Where->getASTContext());
  assert(Matches.size() <= 1 && "Ambiguous match!");
  assert(Matches.size() >= 1 && "Match not found!");
  const T *Node = selectFirst<T>("root", Matches);
  assert(Node && "Type mismatch!");
  return Node;
}

// Find a declaration in the current AST by name.
template <typename T>
const T *findDeclByName(const Decl *Where, StringRef Name) {
  using namespace ast_matchers;
  return findNode<T>(Where, namedDecl(hasName(Name)));
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
      : C(C),
        ChkMgr(C.getASTContext(), *C.getAnalyzerOpts(), C.getPreprocessor()),
        CTU(C), Consumers(),
        AMgr(C.getASTContext(), C.getPreprocessor(), Consumers,
             CreateRegionStoreManager, CreateRangeConstraintManager, &ChkMgr,
             *C.getAnalyzerOpts()),
        VisitedCallees(), FS(),
        Eng(CTU, AMgr, &VisitedCallees, &FS, ExprEngine::Inline_Regular) {}
};

} // namespace ento
} // namespace clang

#endif
