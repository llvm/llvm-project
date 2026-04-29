//===- UnsafeBufferUsageExtractor.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
class UnsafeBufferUsageTUSummaryExtractor : public TUSummaryExtractor {
public:
  UnsafeBufferUsageTUSummaryExtractor(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {}

  Expected<std::unique_ptr<UnsafeBufferUsageEntitySummary>>
  extractEntitySummary(const NamedDecl *Contributor, ASTContext &Ctx);
  void HandleTranslationUnit(ASTContext &Ctx) override;
};
} // namespace clang::ssaf

Expected<std::unique_ptr<UnsafeBufferUsageEntitySummary>>
clang::ssaf::UnsafeBufferUsageTUSummaryExtractor::extractEntitySummary(
    const NamedDecl *Contributor, ASTContext &Ctx) {
  std::set<const Expr *> UnsafePointers;

  auto MatchAction = [&UnsafePointers, &Ctx](const DynTypedNode &Node) {
    matchUnsafePointers(Node, Ctx, UnsafePointers);
  };
  findMatchesIn(Contributor, MatchAction);

  EntityPointerLevelSet Results;

  for (const Expr *Ptr : UnsafePointers) {
    Expected<EntityPointerLevelSet> Translation =
        translateEntityPointerLevel(Ptr, Ctx, [this](const EntityName &EN) {
          return SummaryBuilder.addEntity(EN);
        });

    if (Translation) {
      // Filter out those temporary invalid EntityPointerLevels associated
      // with `&E` pointers. They need no transformation of entities:
      auto FilteredTranslation = llvm::make_filter_range(
          *Translation, [](const EntityPointerLevel &E) -> bool {
            return E.getPointerLevel() > 0;
          });
      Results.insert(FilteredTranslation.begin(), FilteredTranslation.end());
      continue;
    }
    return Translation.takeError();
  }

  return std::make_unique<UnsafeBufferUsageEntitySummary>(
      UnsafeBufferUsageEntitySummary(std::move(Results)));
}

void clang::ssaf::UnsafeBufferUsageTUSummaryExtractor::HandleTranslationUnit(
    ASTContext &Ctx) {
  std::vector<const NamedDecl *> Contributors;

  findContributors(Ctx, Contributors);
  for (auto *CD : Contributors) {
    auto EntitySummary = extractEntitySummary(CD, Ctx);

    if (!EntitySummary)
      llvm::reportFatalInternalError(EntitySummary.takeError());
    assert(*EntitySummary);
    if ((*EntitySummary)->empty())
      continue;

    auto ContributorName = getEntityName(CD);

    if (!ContributorName)
      llvm::reportFatalInternalError(makeEntityNameErr(Ctx, CD));

    [[maybe_unused]] auto [Ignored, InsertionSucceeded] =
        SummaryBuilder.addSummary(SummaryBuilder.addEntity(*ContributorName),
                                  std::move(*EntitySummary));

    assert(InsertionSucceeded && "duplicated contributor extraction");
  }
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageTUSummaryExtractorAnchorSource = 0;

static clang::ssaf::TUSummaryExtractorRegistry::Add<
    ssaf::UnsafeBufferUsageTUSummaryExtractor>
    RegisterExtractor(UnsafeBufferUsageEntitySummary::Name,
                      "The TUSummaryExtractor for unsafe buffer pointers");
