//===- EntityLinker.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/EntityLinker/EntityLinker.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang::ssaf;

//----------------------------------------------------------------------------
// Error Message Constants
//----------------------------------------------------------------------------

namespace {

namespace ErrorMessages {

constexpr const char *EntityIdAlreadyExistsInLinkageTable =
    "EntityId({0}) already exists in LU linkage table";

constexpr const char *FailedToMergeSummaryData =
    "failed to merge summary data for TU EntityId({0}) resolved to LU "
    "EntityId({1}) with linkage '{2}'";

constexpr const char *MissingLinkageInformation =
    "missing linkage information for TU EntityId({0})";

constexpr const char *DuplicateEntityIdInLinking =
    "duplicate TU EntityId({0}) encountered during linking";

constexpr const char *MergingSummaryData = "merging summary data";
constexpr const char *LinkingTUSummary = "linking TU summary";

} // namespace ErrorMessages

} // namespace

static NestedBuildNamespace
resolveNamespace(const NestedBuildNamespace &LUNamespace,
                 const NestedBuildNamespace &EntityNamespace,
                 const EntityLinkage::LinkageType Linkage) {
  switch (Linkage) {
  case EntityLinkage::LinkageType::None:
  case EntityLinkage::LinkageType::Internal:
    return EntityNamespace.makeQualified(LUNamespace);
  case EntityLinkage::LinkageType::External:
    return NestedBuildNamespace(LUNamespace);
  }

  llvm_unreachable("Unhandled EntityLinkage::LinkageType variant");
}

llvm::Expected<EntityId> EntityLinker::resolve(const EntityName &OldName,
                                               const EntityLinkage &Linkage) {
  NestedBuildNamespace NewNamespace = resolveNamespace(
      Output.LUNamespace, OldName.Namespace, Linkage.getLinkage());

  EntityName NewName(OldName.USR, OldName.Suffix, NewNamespace);

  // NewId construction will always return a fresh id for `None` and `Internal`
  // linkage entities since their namespaces will be different even if their
  // names clash. For `External` linkage entities with clashing names this
  // function will return the id assigned at the first insertion.
  EntityId NewId = Output.IdTable.getId(NewName);

  [[maybe_unused]] auto [It, Inserted] =
      Output.LinkageTable.try_emplace(NewId, Linkage);
  // if (!Inserted) {
  //   return ErrorBuilder::create(
  //              llvm::inconvertibleErrorCode(),
  //              ErrorMessages::EntityIdAlreadyExistsInLinkageTable,
  //              NewId.Index)
  //       .build();
  // }

  return NewId;
}

llvm::Error EntityLinker::merge(
    std::map<SummaryName,
             std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
        &InputData,
    std::map<SummaryName,
             std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
        &OutputData,
    const EntityId OldId, const EntityId NewId, const EntityLinkage &Linkage,
    std::vector<EntitySummaryEncoding *> &PatchTargets) {
  for (auto &[Name, DataMap] : InputData) {
    auto Iter = DataMap.find(OldId);
    if (Iter == DataMap.end()) {
      continue;
    }

    auto &OutputMap = OutputData[Name];
    auto InsertResult = OutputMap.insert({NewId, std::move(Iter->second)});

    // If insertion is successful, we will have to replace OldId with NewId in
    // this EntitySummaryEncoding.
    if (InsertResult.second) {
      PatchTargets.push_back(InsertResult.first->second.get());
    } else {
      switch (Linkage.getLinkage()) {
        // Insertion should never fail for `None` and `Internal` linkage
        // entities because these entities have different namespaces even if
        // their names clash.
      case EntityLinkage::LinkageType::None:
      case EntityLinkage::LinkageType::Internal:
        return ErrorBuilder::create(llvm::inconvertibleErrorCode(),
                                    ErrorMessages::FailedToMergeSummaryData,
                                    OldId.Index, NewId.Index,
                                    toString(Linkage.getLinkage()))
            .build();
      case EntityLinkage::LinkageType::External:
        // Insertion is expected to fail for duplicate occurrences of `External`
        // linkage entities. We will report these cases to help users debug
        // potential ODR violations.
        // TODO - issue diagnostic log for dropping data using instrumentation
        // framework.
        break;
      }
    }
  }

  return llvm::Error::success();
}

void EntityLinker::patch(
    std::vector<EntitySummaryEncoding *> &PatchTargets,
    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  for (auto *PatchTarget : PatchTargets) {
    assert(PatchTarget && "Patch target cannot be null");
    PatchTarget->patch(EntityResolutionTable);
  }
}

llvm::Error EntityLinker::link(std::unique_ptr<TUSummaryEncoding> Summary) {
  std::map<EntityId, EntityId> EntityResolutionTable;
  std::vector<EntitySummaryEncoding *> PatchTargets;

  for (const auto &[OldName, OldId] : Summary->IdTable.Entities) {
    auto Iter = Summary->LinkageTable.find(OldId);
    if (Iter == Summary->LinkageTable.end()) {
      return ErrorBuilder::create(llvm::inconvertibleErrorCode(),
                                  ErrorMessages::MissingLinkageInformation,
                                  OldId.Index)
          .context(ErrorMessages::LinkingTUSummary)
          .build();
    }

    const EntityLinkage &Linkage = Iter->second;

    auto NewIdOrErr = resolve(OldName, Linkage);
    if (!NewIdOrErr) {
      return ErrorBuilder::wrap(NewIdOrErr.takeError())
          .context(ErrorMessages::LinkingTUSummary)
          .build();
    }

    EntityId NewId = *NewIdOrErr;

    auto InsertResult = EntityResolutionTable.insert({OldId, NewId});
    if (!InsertResult.second) {
      return ErrorBuilder::create(llvm::inconvertibleErrorCode(),
                                  ErrorMessages::DuplicateEntityIdInLinking,
                                  OldId.Index)
          .context(ErrorMessages::LinkingTUSummary)
          .build();
    }

    if (llvm::Error Err = merge(Summary->Data, Output.Data, OldId, NewId,
                                Linkage, PatchTargets)) {
      return ErrorBuilder::wrap(std::move(Err))
          .context(ErrorMessages::MergingSummaryData)
          .context(ErrorMessages::LinkingTUSummary)
          .build();
    }
  }

  patch(PatchTargets, EntityResolutionTable);

  return llvm::Error::success();
}
