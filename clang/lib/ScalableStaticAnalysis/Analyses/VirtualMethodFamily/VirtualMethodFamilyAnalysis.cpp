//===- VirtualMethodFamilyAnalysis.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <map>
#include <optional>
#include <utility>

using namespace clang::ssaf;

namespace {

struct MethodFamilyUnionFind {
  EntityId find(EntityId E);
  void unionSets(EntityId A, EntityId B);

  auto keys() const { return llvm::make_first_range(Roots); }

private:
  llvm::DenseMap<EntityId, EntityId> Roots;
};

class VirtualMethodFamilyAnalysis final
    : public SummaryAnalysis<VirtualMethodFamilyAnalysisResult,
                             VirtualMethodSummary> {
public:
  llvm::Error add(EntityId Id, const VirtualMethodSummary &Summary) override {
    Data[Id] = &Summary;
    return llvm::Error::success();
  }

  llvm::Error finalize() override;

private:
  /// Fill the \c Family map.
  void groupParamsAndReturnEntities();

  /// Make the param and return IDs share a family.
  void unionParamsAndReturnEntitiesInSummaries(const VirtualMethodSummary &LHS,
                                               const VirtualMethodSummary &RHS);

  MethodFamilyUnionFind Family;
  std::map<EntityId, const VirtualMethodSummary *> Data;
};
} // namespace

EntityId MethodFamilyUnionFind::find(EntityId E) {
  auto It = Roots.find(E);
  if (It == Roots.end()) {
    Roots.try_emplace(E, E); // Self-rooted singleton.
    return E;
  }
  if (It->second == E)
    return E;
  EntityId Root = find(It->second);
  Roots.insert_or_assign(E, Root); // Path compression.
  return Root;
}

void MethodFamilyUnionFind::unionSets(EntityId A, EntityId B) {
  EntityId RootA = find(A);
  EntityId RootB = find(B);
  if (RootA == RootB)
    return;

  // Prefer the lexicographically-smaller rep for stable output across runs.
  if (RootB < RootA)
    std::swap(RootA, RootB);

  Roots.insert_or_assign(RootB, RootA);
}

void VirtualMethodFamilyAnalysis::unionParamsAndReturnEntitiesInSummaries(
    const VirtualMethodSummary &LHS, const VirtualMethodSummary &RHS) {
  assert(LHS.ParamEntities.size() == RHS.ParamEntities.size());
  assert(LHS.ReturnEntity.has_value() == RHS.ReturnEntity.has_value());

  using llvm::zip_equal;
  for (auto [LParam, RParam] : zip_equal(LHS.ParamEntities, RHS.ParamEntities))
    Family.unionSets(LParam, RParam);

  if (LHS.ReturnEntity.has_value())
    Family.unionSets(*LHS.ReturnEntity, *RHS.ReturnEntity);
}

void VirtualMethodFamilyAnalysis::groupParamsAndReturnEntities() {
  for (const VirtualMethodSummary *CurrSum : llvm::make_second_range(Data)) {
    for (EntityId OverriddenMethodId : CurrSum->OverriddenMethods) {
      auto BaseSumIt = Data.find(OverriddenMethodId);
      assert(BaseSumIt != Data.end());
      const VirtualMethodSummary &BaseSum = *BaseSumIt->second;
      unionParamsAndReturnEntitiesInSummaries(*CurrSum, BaseSum);
    }
  }
}

llvm::Error VirtualMethodFamilyAnalysis::finalize() {
  groupParamsAndReturnEntities();

  auto &R = getResult();
  for (EntityId E : Family.keys())
    R.RetAndParamData.insert({E, Family.find(E)});
  return llvm::Error::success();
}

static AnalysisRegistry::Add<VirtualMethodFamilyAnalysis>
    RegisterAnalysis("Override-family equivalence classes for virtual methods");

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

namespace clang::ssaf {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const VirtualMethodFamilyAnalysisResult &R) {
  OS << "VirtualMethodFamilyAnalysisResult with " << R.RetAndParamData.size()
     << " entries ";
  if (R.RetAndParamData.empty())
    return OS << "{}";

  // DenseMap iteration order depends on hashing, so sort for stable output.
  using Entry = std::pair<EntityId, EntityId>;
  llvm::SmallVector<Entry> Entries(R.RetAndParamData.begin(),
                                   R.RetAndParamData.end());
  llvm::sort(Entries,
             [](const Entry &L, const Entry &R) { return L.first < R.first; });

  OS << "{\n";
  for (const auto &[Id, FamilyId] : Entries)
    OS << "  " << Id << " -> " << FamilyId << "\n";
  return OS << "}";
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int VirtualMethodFamilyAnalysisAnchorSource = 0;
} // namespace clang::ssaf
