//===- VirtualMethodFamilyTestSupport.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared fixture for the VirtualMethodFamily tests: parses a snippet, runs the
// VirtualMethod extractor over it, and resolves declarations to the EntityIds
// the extractor minted.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILYTESTSUPPORT_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILYTESTSUPPORT_H

#include "ParsedAST.h"
#include "TestFixture.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "llvm/TargetParser/Triple.h"

#include <map>
#include <optional>
#include <string>

namespace clang::ssaf {

/// Base fixture for tests that need a TUSummary populated by the
/// VirtualMethod extractor.
class VirtualMethodFamilyTestBase : public TestFixture {
protected:
  ParsedAST AST;

  /// Parses \p Code and runs the VirtualMethod extractor over it. Returns
  /// false if the AST could not be built or the extractor is not registered.
  /// Call once per test, before any of the lookups below.
  [[nodiscard]] bool runVirtualMethodExtractor(llvm::StringRef Code) {
    if (!AST.parse(Code))
      return false;
    auto Extractor =
        makeTUSummaryExtractor(VirtualMethodSummary::Name, Builder);
    if (!Extractor)
      return false;
    Extractor->HandleTranslationUnit(AST.getASTContext());
    return true;
  }

  /// Resolves \p ND to the EntityId the extractor minted for it, or
  /// std::nullopt if the extractor produced no entity for it.
  std::optional<EntityId> entityIdOf(const NamedDecl *ND) const {
    return ND ? lookup(getEntityName(ND)) : std::nullopt;
  }

  /// Resolves the return slot of \p FD to its EntityId.
  std::optional<EntityId> returnEntityIdOf(const FunctionDecl *FD) const {
    return FD ? lookup(getEntityNameForReturn(FD)) : std::nullopt;
  }

  /// Resolves an EntityName against the table the extractor populated.
  std::optional<EntityId> lookup(std::optional<EntityName> Name) const {
    if (!Name)
      return std::nullopt;
    const auto &Entities = getEntities(getIdTable(TUSum));
    auto It = Entities.find(*Name);
    if (It == Entities.end())
      return std::nullopt;
    return It->second;
  }

  /// Looks up the extractor's summary for \p FD, or nullptr if it produced
  /// none for this function.
  const VirtualMethodSummary *getMethodSummary(const FunctionDecl *FD) const {
    auto Id = entityIdOf(FD);
    if (!Id)
      return nullptr;
    const auto &Data = getData(TUSum);
    auto SumIt = Data.find(VirtualMethodSummary::summaryName());
    if (SumIt == Data.end())
      return nullptr;
    auto EIt = SumIt->second.find(*Id);
    if (EIt == SumIt->second.end())
      return nullptr;
    return static_cast<const VirtualMethodSummary *>(EIt->second.get());
  }

  /// Count of method-summary entries in the TUSummary.
  std::size_t methodSummaryCount() const {
    const auto &Data = getData(TUSum);
    auto It = Data.find(VirtualMethodSummary::summaryName());
    if (It == Data.end())
      return 0;
    return It->second.size();
  }

  /// Maps every EntityId the extractor minted for the parsed snippet to a
  /// readable label: "Base::foo(int *)", "Base::foo(int *)#return" or
  /// "Base::foo(int *)#param0 'p'". Entities the extractor skipped are absent.
  std::map<EntityId, std::string> entityLabels() const {
    std::map<EntityId, std::string> Labels;
    auto Add = [&](std::optional<EntityId> Id, std::string Label) {
      if (Id)
        Labels.insert({*Id, std::move(Label)});
    };

    for (const FunctionDecl *FD : AST.functions()) {
      std::string Sig = ParsedAST::signatureOf(FD);
      Add(entityIdOf(FD), Sig);
      Add(returnEntityIdOf(FD), Sig + "#return");
      for (unsigned I = 0, E = FD->getNumParams(); I != E; ++I) {
        const ParmVarDecl *P = FD->getParamDecl(I);
        std::string Label = Sig + "#param" + std::to_string(I);
        if (!P->getName().empty())
          Label += " '" + P->getName().str() + "'";
        Add(entityIdOf(P), std::move(Label));
      }
    }
    return Labels;
  }

  /// The entityLabels() mapping as text, to be streamed into a failing
  /// assertion so that the EntityIds in its message can be decoded.
  std::string legend() const {
    std::map<EntityId, std::string> Labels = entityLabels();
    if (Labels.empty())
      return "\nid legend: <no entities extracted>";

    std::string Result;
    llvm::raw_string_ostream OS(Result);
    OS << "\nid legend:";
    for (const auto &[Id, Label] : Labels)
      OS << "\n  " << Id << " = " << Label;
    return Result;
  }

  TUSummary &tuSummary() { return TUSum; }

private:
  SSAFOptions Opts;
  BuildNamespace NS{BuildNamespaceKind::CompilationUnit, "Mock.cpp"};
  TUSummary TUSum{llvm::Triple("arm64-apple-macosx"), NS};
  TUSummaryBuilder Builder{TUSum, Opts};
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_ANALYSES_VIRTUALMETHODFAMILY_VIRTUALMETHODFAMILYTESTSUPPORT_H
