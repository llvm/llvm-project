//===- PointerFlowExtractor.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeBase.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowPairs.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace clang::ssaf {
extern PointerFlowEntitySummary buildPointerFlowEntitySummary(EdgeSet Edges);
} // namespace  clang::ssaf

namespace {
using namespace clang;
using namespace ssaf;

class PointerFlowEdgeBuilder {
public:
  EdgeSet Results;

  PointerFlowEdgeBuilder(ASTContext &Ctx, TUSummaryExtractor &Extractor)
      : Ctx(Ctx), Extractor(Extractor) {}

  llvm::Error operator()(const Expr *LHS, const Expr *RHS);

  llvm::Error operator()(const ValueDecl *LHS, bool IsRet, const Expr *RHS);

private:
  ASTContext &Ctx;
  TUSummaryExtractor &Extractor;

  /// As \c RHS of PointerFlowPairs can still be list-initializers in case
  /// (multi-d) pointer arrays, this function decomposes them recursively and
  /// increases pointer level of \c LHS properly.
  llvm::Error handleRHSAndAddEdges(const DeclPointerLevelVec &LHS,
                                   const Expr *RHS,
                                   unsigned ArrayElementIndirectLevel = 0);

  /// Converts DeclPointerLevelVec pairs to edges:
  llvm::Error addEdges(const DeclPointerLevelVec &LHS,
                       Expected<DeclPointerLevelVec> &&RHS);
};

llvm::Error
PointerFlowEdgeBuilder::addEdges(const DeclPointerLevelVec &LHS,
                                 Expected<DeclPointerLevelVec> &&RHS) {
  if (!RHS)
    return RHS.takeError();
  if (RHS->empty())
    return llvm::Error::success();

  std::vector<DeclPointerLevelVec> LVecs, RVecs;

  LVecs.reserve(LHS.size());
  for (const auto &L : LHS)
    LVecs.push_back(elaborateHigherDeclPointerLevels(L));
  RVecs.reserve(RHS->size());
  for (const auto &R : *RHS)
    RVecs.push_back(elaborateHigherDeclPointerLevels(R));

  // Imagine an assignment from pointer q to p: 'p = q'.  It encodes that if 'p'
  // has some property, so must 'q'; moreover, if '*p/p[i]' has some property,
  // so must '*q/q[i]' and so on.  Therefore, for each edge '(a, n) -> (b, m)'
  // that represents an explicitly spelled place in the source code, we also add
  // '(a, n + 1) -> (b, m + 1)',
  // '(a, n + 2) -> (b, m + 2)', ... continuing until either 'a' or 'b' reaches
  // its maximum pointer level, whichever happens first.
  //
  // Note that type checking ensures that 'p' and 'q' have
  // identical pointer levels, but '(a, n)' and '(b, m)' may have different
  // upper bounds on their pointer levels, when, for example, 'q' is a
  // reinterpret-cast expression, which can have different pointer level than
  // its sub-expression.

  for (const DeclPointerLevelVec &L : LVecs)
    for (const DeclPointerLevelVec &R : RVecs)
      for (const auto &[LDPL, RDPL] : llvm::zip(L, R)) {
        auto LEPL = toEntityPointerLevel(LDPL, Ctx, Extractor);
        if (!LEPL)
          return LEPL.takeError();
        auto REPL = toEntityPointerLevel(RDPL, Ctx, Extractor);
        if (!REPL)
          return REPL.takeError();
        Results[*LEPL].insert(*REPL);
      }
  return llvm::Error::success();
}

llvm::Error
PointerFlowEdgeBuilder::handleRHSAndAddEdges(
    const DeclPointerLevelVec &LHS, const Expr *RHS,
    unsigned ArrayElementIndirectLevel) {
  const auto *ILE = dyn_cast<InitListExpr>(RHS);
  if (!ILE) {
    if (!hasPtrOrArrType(RHS))
      return llvm::Error::success();

    // Leaf: raise a copy of LHS by the array depth reached, then add edges.
    DeclPointerLevelVec Copy = LHS;

    for (DeclPointerLevel &DPL : Copy)
      DPL.PointerLevel += ArrayElementIndirectLevel;
    return addEdges(Copy, translateDeclPointerLevel(RHS, Ctx));
  }

  llvm::Error Err = llvm::Error::success();

  // Descend one array dimension.
  for (const auto *Init : ILE->inits())
    Err = llvm::joinErrors(
        std::move(Err),
        handleRHSAndAddEdges(LHS, Init, ArrayElementIndirectLevel + 1));
  return Err;
}

llvm::Error PointerFlowEdgeBuilder::operator()(const Expr *LHS,
                                               const Expr *RHS) {
  auto LVec = translateDeclPointerLevel(LHS, Ctx);
  if (!LVec)
    return LVec.takeError();
  return handleRHSAndAddEdges(*LVec, RHS);
}

llvm::Error PointerFlowEdgeBuilder::operator()(const ValueDecl *LHS, bool IsRet,
                                               const Expr *RHS) {
  DeclPointerLevelVec LVec = {createDeclPointerLevel(LHS, IsRet)};
  return handleRHSAndAddEdges(LVec, RHS);
}

class PointerFlowTUSummaryExtractor : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

  /// \return a non-null unique pointer to a PointerFlowEntitySummary
  std::unique_ptr<PointerFlowEntitySummary>
  extractEntitySummary(const std::vector<const NamedDecl *> &ContributorDecls,
                       ASTContext &Ctx, TUSummaryExtractor &Extractor) {
    ssaf::PointerFlowPairMatcher Matcher(Ctx);
    PointerFlowEdgeBuilder Builder(Ctx, Extractor);

    for (const auto *Contrib : ContributorDecls) {
      auto MatchAction = [&](const DynTypedNode &Node) {
        llvm::SmallVector<PointerFlowPair> Pairs;

        Matcher.matches(Node, Contrib, Pairs);
        for (auto &Pair : Pairs)
          if (auto Err = Pair.visitLHS(Builder, Pair.RHS))
            logWarningFromError(std::move(Err));
      };

      findMatchesIn(Contrib, MatchAction);
    }
    return std::make_unique<PointerFlowEntitySummary>(
        buildPointerFlowEntitySummary(std::move(Builder.Results)));
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    extractAndAddSummaries(
        *this, SummaryBuilder, Ctx,
        [&](const std::vector<const NamedDecl *> &Decls) {
          return extractEntitySummary(Decls, Ctx, *this);
        },
        "PointerFlow");
  }
};
} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int PointerFlowExtractorAnchorSource = 0;
} // namespace clang::ssaf

static TUSummaryExtractorRegistry::Add<PointerFlowTUSummaryExtractor>
    RegisterExtractor(PointerFlowEntitySummary::Name,
                      "Extract pointer flow information");
