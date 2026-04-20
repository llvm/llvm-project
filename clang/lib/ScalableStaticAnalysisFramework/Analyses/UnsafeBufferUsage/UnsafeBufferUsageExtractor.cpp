//===- UnsafeBufferUsageExtractor.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageExtractor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace {
using namespace clang;
using namespace ssaf;

static llvm::Error makeCreateEntityNameError(const NamedDecl *FailedDecl,
                                             ASTContext &Ctx) {
  std::string LocStr = FailedDecl->getSourceRange().getBegin().printToString(
      Ctx.getSourceManager());
  return llvm::createStringError(
      "failed to create entity name for %s declared at %s",
      FailedDecl->getNameAsString().c_str(), LocStr.c_str());
}

static llvm::Error makeAddEntitySummaryError(const NamedDecl *FailedContributor,
                                             ASTContext &Ctx) {
  std::string LocStr =
      FailedContributor->getSourceRange().getBegin().printToString(
          Ctx.getSourceManager());
  return llvm::createStringError(
      "failed to add entity summary for contributor %s declared at %s",
      FailedContributor->getNameAsString().c_str(), LocStr.c_str());
}

Expected<EntityPointerLevelSet>
buildEntityPointerLevels(std::set<const Expr *> &&UnsafePointers,
                         UnsafeBufferUsageTUSummaryExtractor &Extractor,
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
} // namespace

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

std::unique_ptr<UnsafeBufferUsageEntitySummary>
UnsafeBufferUsageTUSummaryExtractor::extractEntitySummary(
    const Decl *Contributor, ASTContext &Ctx, llvm::Error &Error) {
  auto AddEntity = [this](EntityName EN) { return addEntity(EN); };
  Expected<EntityPointerLevelSet> EPLs = buildEntityPointerLevels(
      findUnsafePointersInContributor(Contributor), *this, Ctx, AddEntity);

  if (EPLs)
    return std::make_unique<UnsafeBufferUsageEntitySummary>(
        UnsafeBufferUsageEntitySummary(std::move(*EPLs)));
  Error = EPLs.takeError();
  return nullptr;
}

void UnsafeBufferUsageTUSummaryExtractor::HandleTranslationUnit(
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
      return false;
    }
  } ContributorFinder;

  ContributorFinder.VisitTranslationUnitDecl(Ctx.getTranslationUnitDecl());

  llvm::Error Errors = llvm::ErrorSuccess();
  auto addError = [&Errors](llvm::Error Err) {
    Errors = llvm::joinErrors(std::move(Errors), std::move(Err));
  };

  for (auto *CD : ContributorFinder.Contributors) {
    llvm::Error Error = llvm::ErrorSuccess();
    auto EntitySummary = extractEntitySummary(CD, Ctx, Error);

    if (Error)
      addError(std::move(Error));
    if (EntitySummary->empty())
      continue;

    auto ContributorName = getEntityName(CD);

    if (!ContributorName) {
      addError(makeCreateEntityNameError(CD, Ctx));
      continue;
    }

    auto [EntitySummaryPtr, Success] = SummaryBuilder.addSummary(
        addEntity(*ContributorName), std::move(EntitySummary));

    if (!Success)
      addError(makeAddEntitySummaryError(CD, Ctx));
  }
  // FIXME: handle errors!
  llvm::consumeError(std::move(Errors));
}
