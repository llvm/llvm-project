//===- UnsafeBufferUsageExtractor.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace {
using namespace clang;
using namespace ssaf;

llvm::Error makeCreateEntityNameError(const NamedDecl *FailedDecl,
                                      ASTContext &Ctx) {
  std::string LocStr = FailedDecl->getSourceRange().getBegin().printToString(
      Ctx.getSourceManager());
  return llvm::createStringError(
      "failed to create entity name for %s declared at %s",
      FailedDecl->getNameAsString().c_str(), LocStr.c_str());
}

Expected<EntityPointerLevelSet>
buildEntityPointerLevels(std::set<const Expr *> &&UnsafePointers,
                         ASTContext &Ctx,
                         std::function<EntityId(EntityName)> AddEntity) {
  EntityPointerLevelSet Result{};
  llvm::Error AllErrors = llvm::ErrorSuccess();

  for (const Expr *Ptr : UnsafePointers) {
    Expected<EntityPointerLevelSet> Translation =
        translateEntityPointerLevel(Ptr, Ctx, AddEntity);

    if (Translation) {
      // Filter out those temporary invalid EntityPointerLevels associated with
      // `&E` pointers:
      auto FilteredTranslation = llvm::make_filter_range(
          *Translation, [](const EntityPointerLevel &E) -> bool {
            return E.getPointerLevel() > 0;
          });
      Result.insert(FilteredTranslation.begin(), FilteredTranslation.end());
      continue;
    }
    AllErrors = llvm::joinErrors(std::move(AllErrors), Translation.takeError());
  }
  if (AllErrors)
    return AllErrors;
  return Result;
}

static std::set<const Expr *> findUnsafePointersInContributor(const Decl *D) {
  if (isa<FunctionDecl>(D) || isa<VarDecl>(D))
    return findUnsafePointers(D);
  if (auto *RD = dyn_cast<RecordDecl>(D)) {
    std::set<const Expr *> Result;

    for (const FieldDecl *FD : RD->fields()) {
      Result.merge(findUnsafePointers(FD));
    }
    return Result;
  }
  return {};
}
} // namespace

namespace clang::ssaf {
class UnsafeBufferUsageTUSummaryExtractor : public TUSummaryExtractor {
public:
  UnsafeBufferUsageTUSummaryExtractor(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {}

  EntityId addEntity(EntityName EN) { return SummaryBuilder.addEntity(EN); }

  Expected<std::unique_ptr<UnsafeBufferUsageEntitySummary>>
  extractEntitySummary(const Decl *Contributor, ASTContext &Ctx);

  void HandleTranslationUnit(ASTContext &Ctx) override;
};
} // namespace clang::ssaf

Expected<std::unique_ptr<UnsafeBufferUsageEntitySummary>>
clang::ssaf::UnsafeBufferUsageTUSummaryExtractor::extractEntitySummary(
    const Decl *Contributor, ASTContext &Ctx) {
  auto AddEntity = [this](EntityName EN) { return addEntity(EN); };
  Expected<EntityPointerLevelSet> EPLs = buildEntityPointerLevels(
      findUnsafePointersInContributor(Contributor), Ctx, AddEntity);

  if (EPLs)
    return std::make_unique<UnsafeBufferUsageEntitySummary>(
        UnsafeBufferUsageEntitySummary(std::move(*EPLs)));
  return EPLs.takeError();
}

void clang::ssaf::UnsafeBufferUsageTUSummaryExtractor::HandleTranslationUnit(
    ASTContext &Ctx) {

  // FIXME: I suppose finding contributor Decls is commonly needed by all/many
  // extractors
  class ContributorFinder : public DynamicRecursiveASTVisitor {
  public:
    std::vector<const NamedDecl *> Contributors;

    bool VisitFunctionDecl(FunctionDecl *D) override {
      Contributors.push_back(D);
      return true;
    }

    bool VisitRecordDecl(RecordDecl *D) override {
      Contributors.push_back(D);
      return true;
    }

    bool VisitVarDecl(VarDecl *D) override {
      DeclContext *DC = D->getDeclContext();

      if (DC->isFileContext() || DC->isNamespace())
        Contributors.push_back(D);
      return true;
    }
  } ContributorFinder;

  ContributorFinder.TraverseAST(Ctx);
  for (auto *CD : ContributorFinder.Contributors) {
    auto EntitySummary = extractEntitySummary(CD, Ctx);

    if (!EntitySummary)
      llvm::reportFatalInternalError(EntitySummary.takeError());
    assert(*EntitySummary);
    if ((*EntitySummary)->empty())
      continue;

    auto ContributorName = getEntityName(CD);

    if (!ContributorName)
      llvm::reportFatalInternalError(makeCreateEntityNameError(CD, Ctx));

    auto [Ignored, InsertionSucceeded] = SummaryBuilder.addSummary(
        addEntity(*ContributorName), std::move(*EntitySummary));

    assert(InsertionSucceeded && "duplicated contributor extraction");
  }
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageTUSummaryExtractorAnchorSource = 0;

static clang::ssaf::TUSummaryExtractorRegistry::Add<
    ssaf::UnsafeBufferUsageTUSummaryExtractor>
    RegisterExtractor(UnsafeBufferUsageEntitySummary::Name,
                      "The TUSummaryExtractor for unsafe buffer pointers");
