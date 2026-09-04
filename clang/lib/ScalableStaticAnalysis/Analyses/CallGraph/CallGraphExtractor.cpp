//===- CallGraphExtractor.cpp - Call Graph Summary Extractor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/SourceManager.h"
#include "clang/ScalableStaticAnalysis/Analyses/CallGraph/CallGraphSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace {
class CallGraphExtractor final : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

private:
  void HandleTranslationUnit(ASTContext &Ctx) override;

  void handleCallGraphNode(const ASTContext &Ctx, const CallGraphNode *N);
};
} // namespace

void CallGraphExtractor::HandleTranslationUnit(ASTContext &Ctx) {
  // FIXME: Depending on the IncludeLocalEntities option, the extractor should
  // include or exclude calls to function-local defined:
  //   - lambda functions
  //   - methods of local classes
  // Currently, the extractor always includes these callees, even if
  // IncludeLocalEntities is false.
  CallGraph CG;
  CG.addToCallGraph(
      const_cast<TranslationUnitDecl *>(Ctx.getTranslationUnitDecl()));

  for (const auto &N : llvm::make_second_range(CG)) {
    if (N && N->getDecl() && N->getDefinition())
      handleCallGraphNode(Ctx, N.get());
  }
}

void CallGraphExtractor::handleCallGraphNode(const ASTContext &Ctx,
                                             const CallGraphNode *N) {
  const FunctionDecl *Definition = N->getDefinition();

  // FIXME: `clang::CallGraph` does not create entries for primary templates.
  assert(!Definition->isTemplated());

  auto CallerId = addEntity(Definition);
  if (!CallerId)
    return;

  auto FnSummary = std::make_unique<CallGraphSummary>();

  PresumedLoc Loc =
      Ctx.getSourceManager().getPresumedLoc(Definition->getLocation());
  FnSummary->Definition.File = Loc.getFilename();
  FnSummary->Definition.Line = Loc.getLine();
  FnSummary->Definition.Column = Loc.getColumn();
  FnSummary->PrettyName = AnalysisDeclContext::getFunctionName(Definition);

  for (const auto &Record : N->callees()) {
    const Decl *CalleeDecl = Record.Callee->getDecl();

    // FIXME: `clang::CallGraph` does not consider indirect calls, thus this is
    // never null.
    assert(CalleeDecl);

    // `clang::CallGraph` resolves ObjCMessageExprs (including property
    // dot-syntax) to their ObjCMethodDecls and adds them as callees — see
    // `CGBuilder::VisitObjCMessageExpr` in clang/lib/Analysis/CallGraph.cpp.
    // ObjC dispatch is dynamic, so recording these as direct callees would be
    // misleading; skip them until we model ObjC properly.
    if (isa<ObjCMethodDecl>(CalleeDecl))
      continue;

    // FIXME: `clang::CallGraph` does not create entries for primary templates.
    assert(!CalleeDecl->isTemplated());

    auto CalleeId = addEntity(cast<NamedDecl>(CalleeDecl));
    if (!CalleeId)
      continue;

    if (const auto *MD = dyn_cast_or_null<CXXMethodDecl>(CalleeDecl);
        MD && MD->isVirtual()) {
      FnSummary->VirtualCallees.insert(*CalleeId);
      continue;
    }
    FnSummary->DirectCallees.insert(*CalleeId);
  }

  SummaryBuilder.addSummary(*CallerId, std::move(FnSummary));
}

static TUSummaryExtractorRegistry::Add<CallGraphExtractor>
    RegisterExtractor(CallGraphSummary::Name,
                      "Extracts static call-graph information");

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CallGraphExtractorAnchorSource = 0;
} // namespace clang::ssaf
