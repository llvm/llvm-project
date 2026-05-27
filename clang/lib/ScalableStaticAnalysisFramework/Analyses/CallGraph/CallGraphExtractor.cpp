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
#include "clang/ScalableStaticAnalysisFramework/Analyses/CallGraph/CallGraphSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
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

    // FIXME: `clang::CallGraph` does not consider ObjCMessageExprs as calls.
    // Consequently, they don't appear as a Callee.
    assert(!isa<ObjCMethodDecl>(CalleeDecl));

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

// This anchor is used to force the linker to link in the generated object file
// and thus register the CallGraphExtractor.
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CallGraphExtractorAnchorSource = 0;
