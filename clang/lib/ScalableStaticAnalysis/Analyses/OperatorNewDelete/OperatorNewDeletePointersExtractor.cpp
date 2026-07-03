//===- OperatorNewDeletePointersExtractor.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extractor implementation for extracting from user-provided operator
// new/delete overloadings:
//  1 return entities of operator new overloads;
//  2 the parameter (optionally the 2nd)  of operator new overloads
//    representing the pointer to a memory area to initialize the object at;
//  3 the first parameter of operator delete overloads representing the pointer
//    to a memory block to deallocate or a null pointer;
//  4 the parameter (optionally the 2nd)  of operator delete overloads
//    representing the pointer used as the placement parameter in the matching
//    placement new.
//
//===----------------------------------------------------------------------===//

#include "../SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/ScalableStaticAnalysis/Analyses/OperatorNewDelete/OperatorNewDeletePointers.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace {

class OperatorNewDeletePointersExtractor final : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

private:
  void HandleTranslationUnit(ASTContext &Ctx) override;

  std::unique_ptr<OperatorNewDeletePointersEntitySummary>
  extractEntitySummary(const std::vector<const NamedDecl *> &Decls);
};

void OperatorNewDeletePointersExtractor::HandleTranslationUnit(
    ASTContext &Ctx) {
  extractAndAddSummaries(
      *this, SummaryBuilder, Ctx,
      [&](const std::vector<const NamedDecl *> &Decls) {
        return extractEntitySummary(Decls);
      },
      OperatorNewDeletePointersEntitySummary::Name);
}

std::unique_ptr<OperatorNewDeletePointersEntitySummary>
OperatorNewDeletePointersExtractor::extractEntitySummary(
    const std::vector<const NamedDecl *> &ContributorDecls) {
  auto Summary = std::make_unique<OperatorNewDeletePointersEntitySummary>();
  auto Matcher = [&Summary, this](const DynTypedNode &Node) {
    const auto *FD = Node.get<FunctionDecl>();

    if (!FD)
      return;

    OverloadedOperatorKind OO = FD->getOverloadedOperator();

    switch (OO) {
    case OO_New:
    case OO_Array_New:
      // Extract case 1:
      if (auto Id = addEntityForReturn(FD))
        Summary->Entities.insert(*Id);
      break;
    case OO_Delete:
    case OO_Array_Delete:
      // Extract case 3; ignore ill-formed ones (first param not a pointer).
      if (!FD->getNumParams() || !hasPtrOrArrType(FD->getParamDecl(0)))
        return;
      if (auto Id = addEntity(FD->getParamDecl(0)))
        Summary->Entities.insert(*Id);
      break;
    default:
      return;
    };
    // Extract case 2 & 4: only `operator new(size_t, void*)` and
    // `operator delete(void*, void*)` are standard-defined with a void* 2nd
    // param; for user-defined 3+ param overloads the 2nd param type is
    // unconstrained, so we conservatively skip them.
    if (FD->getNumParams() == 2 && hasPtrOrArrType(FD->getParamDecl(1))) {
      if (auto Id = addEntity(FD->getParamDecl(1)))
        Summary->Entities.insert(*Id);
    }
  };

  for (const NamedDecl *Decl : ContributorDecls)
    findMatchesIn(Decl, Matcher);
  return Summary;
}

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int OperatorNewDeletePointersExtractorAnchorSource = 0;
} // namespace clang::ssaf

static TUSummaryExtractorRegistry::Add<OperatorNewDeletePointersExtractor>
    RegisterOperatorNewDeletePointersExtractor(
        OperatorNewDeletePointersEntitySummary::Name,
        "Extract pointer entities in operator new/delete overloads that must "
        "have a 'void*' type");
