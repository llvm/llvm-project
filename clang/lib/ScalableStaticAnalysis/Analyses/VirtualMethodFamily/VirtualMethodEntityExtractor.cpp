//===- VirtualMethodEntityExtractor.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extract what virtual methods override what other methods.
// The parameters might be also important for consumers so collect those as
// well - alongside with the ID of the return value.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include <memory>
#include <optional>

using namespace clang;
using namespace ssaf;

namespace {

class VirtualMethodEntityExtractor final : public TUSummaryExtractor,
                                           ConstDynamicRecursiveASTVisitor {
public:
  explicit VirtualMethodEntityExtractor(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {
    ShouldVisitTemplateInstantiations = true;
    ShouldWalkTypesOfTypeLocs = false;
    ShouldVisitImplicitCode = false;
    ShouldVisitLambdaBody = true;
  }

private:
  void HandleTranslationUnit(ASTContext &Ctx) override { TraverseAST(Ctx); }

  bool VisitCXXMethodDecl(const CXXMethodDecl *MD) override;
};
} // namespace

bool VirtualMethodEntityExtractor::VisitCXXMethodDecl(const CXXMethodDecl *MD) {
  if (!MD->isVirtual())
    return true;

  std::optional<EntityId> MethodId = addEntity(MD);
  if (!MethodId)
    return true;

  auto Summary = std::make_unique<VirtualMethodSummary>();
  Summary->ParamEntities.reserve(MD->getNumParams());

  for (const ParmVarDecl *P : MD->parameters()) {
    auto ParamId = addEntity(P);
    if (!ParamId) {
      // If we can't get an EntityId for a parameter, drop the entire summary
      // rather than leaving a half-populated record.
      return true;
    }
    Summary->ParamEntities.push_back(ParamId.value());
  }

  if (auto ReturnId = addEntityForReturn(MD))
    Summary->ReturnEntity = ReturnId.value();

  for (const CXXMethodDecl *Overridden : MD->overridden_methods()) {
    // We may not be able to convert methods that are coming from system
    // headers, so skip them gracefully.
    if (auto OverriddenId = addEntity(Overridden))
      Summary->OverriddenMethods.push_back(*OverriddenId);
  }

  SummaryBuilder.addSummary(MethodId.value(), std::move(Summary));
  return true;
}

static TUSummaryExtractorRegistry::Add<VirtualMethodEntityExtractor>
    RegisterExtractor(VirtualMethodSummary::Name,
                      "Extract information about virtual methods");

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int VirtualMethodEntityExtractorAnchorSource = 0;
} // namespace clang::ssaf
