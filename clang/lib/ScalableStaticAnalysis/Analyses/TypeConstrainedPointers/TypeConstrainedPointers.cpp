//===- TypeConstrainedPointers.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Type Constrained Pointers analysis, including
//
// a) the Extractor implementation collecting pointer entities that shall retain
//    their types
//
// b) the WPA implementation that simply groups extracted summaries into a
//    TypeConstrainedPointersAnalysisResult that other analysis can use.
//
// c) serialization implementations
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/TypeConstrainedPointers/TypeConstrainedPointers.h"
#include "../SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace {

// Extracts pointer entities that must retain their pointer type.
//
// From `operator new` / `operator delete` overloads:
//    1 the return entity of `operator new` overloads;
//    2 the second parameter of `operator new(size_t, void*)` representing the
//      pointer to the memory area at which to initialize the object;
//    3 the first parameter of `operator delete` overloads representing the
//      pointer to the memory block to deallocate (or a null pointer);
//    4 the second parameter of `operator delete(void*, void*)` representing
//      the placement pointer matching the corresponding placement `new`.
//
// From the `main` function:
//    5 pointer-typed parameters of `main`.
class TypeConstrainedPointersExtractor final : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

private:
  void HandleTranslationUnit(ASTContext &Ctx) override;

  std::unique_ptr<TypeConstrainedPointersEntitySummary>
  extractEntitySummary(const std::vector<const NamedDecl *> &Decls);
};

void TypeConstrainedPointersExtractor::HandleTranslationUnit(ASTContext &Ctx) {
  extractAndAddSummaries(
      *this, SummaryBuilder, Ctx,
      [&](const std::vector<const NamedDecl *> &Decls) {
        return extractEntitySummary(Decls);
      },
      TypeConstrainedPointersEntitySummary::Name);
}

std::unique_ptr<TypeConstrainedPointersEntitySummary>
TypeConstrainedPointersExtractor::extractEntitySummary(
    const std::vector<const NamedDecl *> &ContributorDecls) {
  auto Summary = std::make_unique<TypeConstrainedPointersEntitySummary>();
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
      // Extract case 5: pointer-typed parameters of main.
      if (FD->isMain())
        for (const ParmVarDecl *PVD : FD->parameters()) {
          if (hasPtrOrArrType(PVD))
            if (auto Id = addEntity(PVD))
              Summary->Entities.insert(*Id);
        }
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

//===----------------------------------------------------------------------===//
//                        WPA implementation
//===----------------------------------------------------------------------===//
class TypeConstrainedPointersAnalysis final
    : public SummaryAnalysis<TypeConstrainedPointersAnalysisResult,
                             TypeConstrainedPointersEntitySummary> {
public:
  llvm::Error
  add(EntityId, const TypeConstrainedPointersEntitySummary &Summary) override {
    getResult().Entities.insert(Summary.Entities.begin(),
                                Summary.Entities.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<TypeConstrainedPointersAnalysis>
    RegisterTypeConstrainedPointersAnalysis(
        "Whole-program set of pointer entities that "
        "must retain their pointer type");

//===----------------------------------------------------------------------===//
//                        serialization implementation
//===----------------------------------------------------------------------===//

llvm::json::Object serializeImpl(const std::set<EntityId> &Set,
                                 JSONFormat::EntityIdToJSONFn Fn) {
  llvm::json::Object Result;
  llvm::json::Array DataArray;

  DataArray.reserve(Set.size());
  for (const auto &Ent : Set)
    DataArray.push_back(Fn(Ent));
  Result[TypeConstrainedPointersAnalysisResult::Name] = std::move(DataArray);
  return Result;
}

llvm::Expected<std::set<EntityId>>
deserializeImpl(const llvm::json::Object &Data,
                JSONFormat::EntityIdFromJSONFn Fn) {
  const auto *DataArray =
      Data.getArray(TypeConstrainedPointersAnalysisResult::Name);
  std::set<EntityId> EntitySet;

  if (!DataArray) {
    return makeSawButExpectedError(
        Data, "An object with a key %s",
        TypeConstrainedPointersAnalysisResult::Name.data());
  }
  for (const auto &Elt : *DataArray) {
    const auto *EltAsObj = Elt.getAsObject();

    if (!EltAsObj)
      return makeSawButExpectedError(Elt, "an object representing EntityId");

    Expected<EntityId> EntityId = Fn(*EltAsObj);

    if (!EntityId)
      return EntityId.takeError();

    EntitySet.insert(*EntityId);
  }
  return EntitySet;
}

llvm::json::Object serializeSummary(const EntitySummary &S,
                                    JSONFormat::EntityIdToJSONFn Fn) {
  const auto &SS = static_cast<const TypeConstrainedPointersEntitySummary &>(S);
  return serializeImpl(SS.Entities, Fn);
}

llvm::Expected<std::unique_ptr<EntitySummary>>
deserializeSummary(const llvm::json::Object &Data, EntityIdTable &,
                   JSONFormat::EntityIdFromJSONFn Fn) {
  llvm::Expected<std::set<EntityId>> EntityIDSet = deserializeImpl(Data, Fn);

  if (!EntityIDSet)
    return EntityIDSet.takeError();

  std::unique_ptr<TypeConstrainedPointersEntitySummary> Sum =
      std::make_unique<TypeConstrainedPointersEntitySummary>();

  Sum->Entities = std::move(*EntityIDSet);
  return Sum;
}

struct TypeConstrainedPointersJSONFormatInfo final : JSONFormat::FormatInfo {
  TypeConstrainedPointersJSONFormatInfo()
      : JSONFormat::FormatInfo(
            TypeConstrainedPointersEntitySummary::summaryName(),
            serializeSummary, deserializeSummary) {}
};

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    TypeConstrainedPointersJSONFormatInfo>
    RegisterTypeConstrainedPointersJSONFormatInfo(
        TypeConstrainedPointersEntitySummary::Name,
        "JSON Format info for TypeConstrainedPointersEntitySummary");

llvm::json::Object
serializeAnalysisResult(const TypeConstrainedPointersAnalysisResult &R,
                        JSONFormat::EntityIdToJSONFn Fn) {
  return serializeImpl(R.Entities, Fn);
}

llvm::Expected<std::unique_ptr<AnalysisResult>>
deserializeAnalysisResult(const llvm::json::Object &Data,
                          JSONFormat::EntityIdFromJSONFn Fn) {
  llvm::Expected<std::set<EntityId>> EntityIDSet = deserializeImpl(Data, Fn);

  if (!EntityIDSet)
    return EntityIDSet.takeError();

  std::unique_ptr<TypeConstrainedPointersAnalysisResult> AR =
      std::make_unique<TypeConstrainedPointersAnalysisResult>();

  AR->Entities = std::move(*EntityIDSet);
  return AR;
}

JSONFormat::AnalysisResultRegistry::Add<TypeConstrainedPointersAnalysisResult>
    RegisterTypeConstrainedPointersAnalysisResultForJSON(
        serializeAnalysisResult, deserializeAnalysisResult);

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int TypeConstrainedPointersAnchorSource = 0;
} // namespace clang::ssaf

static TUSummaryExtractorRegistry::Add<TypeConstrainedPointersExtractor>
    RegisterTypeConstrainedPointersExtractor(
        TypeConstrainedPointersEntitySummary::Name,
        "Extract pointer entities that must retain their pointer type");
