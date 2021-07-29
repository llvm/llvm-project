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
#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "gtest/gtest.h"

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

struct ExpectedLocationTy {
  unsigned Line;
  unsigned Column;

  void testEquality(SourceLocation L, SourceManager &SM) const {
    EXPECT_EQ(SM.getSpellingLineNumber(L), Line);
    EXPECT_EQ(SM.getSpellingColumnNumber(L), Column);
  }
};

struct ExpectedRangeTy {
  ExpectedLocationTy Begin;
  ExpectedLocationTy End;

  void testEquality(SourceRange R, SourceManager &SM) const {
    Begin.testEquality(R.getBegin(), SM);
    End.testEquality(R.getEnd(), SM);
  }
};

struct ExpectedPieceTy {
  ExpectedLocationTy Loc;
  std::string Text;
  std::vector<ExpectedRangeTy> Ranges;

  void testEquality(const PathDiagnosticPiece &Piece, SourceManager &SM) {
    Loc.testEquality(Piece.getLocation().asLocation(), SM);
    EXPECT_EQ(Piece.getString(), Text);
    EXPECT_EQ(Ranges.size(), Piece.getRanges().size());
    for (const auto &RangeItem : llvm::enumerate(Piece.getRanges()))
      Ranges[RangeItem.index()].testEquality(RangeItem.value(), SM);
  }
};

struct ExpectedDiagTy {
  ExpectedLocationTy Loc;
  std::string VerboseDescription;
  std::string ShortDescription;
  std::string CheckerName;
  std::string BugType;
  std::string Category;
  std::vector<ExpectedPieceTy> Path;

  void testEquality(const PathDiagnostic &Diag, SourceManager &SM) {
    Loc.testEquality(Diag.getLocation().asLocation(), SM);
    EXPECT_EQ(Diag.getVerboseDescription(), VerboseDescription);
    EXPECT_EQ(Diag.getShortDescription(), ShortDescription);
    EXPECT_EQ(Diag.getCheckerName(), CheckerName);
    EXPECT_EQ(Diag.getBugType(), BugType);
    EXPECT_EQ(Diag.getCategory(), Category);

    EXPECT_EQ(Path.size(), Diag.path.size());
    for (const auto &PieceItem : llvm::enumerate(Diag.path)) {
      if (PieceItem.index() < Path.size())
        Path[PieceItem.index()].testEquality(*PieceItem.value(), SM);
    }
  }
};

using ExpectedDiagsTy = std::vector<ExpectedDiagTy>;

// A consumer to verify the generated diagnostics.
class VerifyPathDiagnosticConsumer : public PathDiagnosticConsumer {
  ExpectedDiagsTy ExpectedDiags;
  SourceManager &SM;

public:
  VerifyPathDiagnosticConsumer(ExpectedDiagsTy &&ExpectedDiags,
                               SourceManager &SM)
      : ExpectedDiags(ExpectedDiags), SM(SM) {}

  StringRef getName() const override { return "verify test diagnostics"; }

  void FlushDiagnosticsImpl(std::vector<const PathDiagnostic *> &Diags,
                            FilesMade *filesMade) override {
    EXPECT_EQ(Diags.size(), ExpectedDiags.size());
    for (const auto &Item : llvm::enumerate(Diags))
      if (Item.index() < ExpectedDiags.size())
        ExpectedDiags[Item.index()].testEquality(*Item.value(), SM);
  }
};

} // namespace ento
} // namespace clang

#endif
