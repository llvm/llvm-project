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
#include <cassert>
#include <system_error>

using namespace clang::ssaf;

//----------------------------------------------------------------------------
// Error Message Constants
//----------------------------------------------------------------------------

namespace {

namespace ErrorMessages {

constexpr const char *EntityAlreadyExistsInLinkageTable =
    "EntityLinker: Entity {0} with {1} linkage already exists in "
    "LinkageTable - indicates corrupted data or logic bug";

constexpr const char *MissingLinkageInformation =
    "EntityLinker: Entity {0} is missing linkage "
    "information in TU summary - indicates corrupted TUSummary";

constexpr const char *DuplicateEntityIdInTUSummary =
    "EntityLinker: Duplicate entity ID {0} in TU summary - indicates "
    "corrupted TUSummary with duplicate entities";

constexpr const char *EntityNotFoundInResolutionTable =
    "EntityLinker: Entity {0} not found in EntityResolutionTable - "
    "indicates corrupted TUSummary or bug in resolve logic";

constexpr const char *FailedToInsertEntityIntoOutputSummary =
    "EntityLinker: Failed to insert data against SummaryName({0}) for "
    "EntityId({0}) with linkage '{1}' into - indicates corrupted data or logic "
    "bug";

constexpr const char *DuplicateTUNamespace =
    "failed to link TU summary: duplicate namespace '{0}'";

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

EntityId EntityLinker::resolveEntity(const EntityName &OldName,
                                     const EntityLinkage &Linkage) {
  NestedBuildNamespace NewNamespace = resolveNamespace(
      Output.LUNamespace, OldName.Namespace, Linkage.getLinkage());

  EntityName NewName(OldName.USR, OldName.Suffix, NewNamespace);

  // NewId construction will always return a fresh id for `None` and `Internal`
  // linkage entities since their namespaces will be different even if their
  // names clash. For `External` linkage entities with clashing names this
  // function will return the id assigned at the first insertion.
  EntityId NewId = Output.IdTable.getId(NewName);

  auto InsertResult = Output.LinkageTable.try_emplace(NewId, Linkage);
  if (!InsertResult.second) {
    // Insertion failure for None/Internal linkage is a fatal error because
    // these entities have unique namespaces and should never collide.
    // External linkage entities may collide (expected for duplicate
    // definitions).
    if (Linkage.getLinkage() == EntityLinkage::LinkageType::None ||
        Linkage.getLinkage() == EntityLinkage::LinkageType::Internal) {
      ErrorBuilder::fatal(ErrorMessages::EntityAlreadyExistsInLinkageTable,
                          NewId.Index, toString(Linkage.getLinkage()));
    }
  }

  return NewId;
}

std::map<EntityId, EntityId> EntityLinker::resolve(TUSummaryEncoding &Summary) {
  std::map<EntityId, EntityId> EntityResolutionTable;

  for (const auto &[OldName, OldId] : Summary.IdTable.Entities) {
    auto Iter = Summary.LinkageTable.find(OldId);
    if (Iter == Summary.LinkageTable.end()) {
      ErrorBuilder::fatal(ErrorMessages::MissingLinkageInformation,
                          OldId.Index);
    }

    const EntityLinkage &Linkage = Iter->second;

    EntityId NewId = resolveEntity(OldName, Linkage);

    auto InsertResult = EntityResolutionTable.insert({OldId, NewId});
    if (!InsertResult.second) {
      ErrorBuilder::fatal(ErrorMessages::DuplicateEntityIdInTUSummary,
                          OldId.Index);
    }
  }

  return EntityResolutionTable;
}

std::vector<EntitySummaryEncoding *>
EntityLinker::merge(TUSummaryEncoding &Summary,
                    std::map<EntityId, EntityId> EntityResolutionTable) {
  std::vector<EntitySummaryEncoding *> PatchTargets;

  for (auto &[SN, DataMap] : Summary.Data) {
    auto &OutputSummaryData = Output.Data[SN];

    for (auto &[OldId, ES] : DataMap) {

      auto Iter = EntityResolutionTable.find(OldId);
      if (Iter == EntityResolutionTable.end()) {
        ErrorBuilder::fatal(ErrorMessages::EntityNotFoundInResolutionTable,
                            OldId.Index);
      }
      const auto NewId = Iter->second;
      auto InsertionResult =
          OutputSummaryData.try_emplace(NewId, std::move(ES));

      if (InsertionResult.second) {
        PatchTargets.push_back(InsertionResult.first->second.get());
      } else {
        // Safe to retrieve linkage using .at since the resolve step ensures
        // linkage information is always present for every Id.
        auto LinkageType = Summary.LinkageTable.at(OldId).getLinkage();

        // Insertion should never fail for `None` and `Internal` linkage
        // entities because these entities have different namespaces even if
        // their names clash.
        if (LinkageType == EntityLinkage::LinkageType::None ||
            LinkageType == EntityLinkage::LinkageType::Internal) {
          ErrorBuilder::fatal(
              ErrorMessages::FailedToInsertEntityIntoOutputSummary, NewId.Index,
              toString(LinkageType));
        }

        // Insertion is expected to fail for duplicate occurrences of
        // `External` linkage entities.
        // TODO - report these cases in a "debug" mode to help
        // debug potential ODR violations.
      }
    }
  }

  return PatchTargets;
}

void EntityLinker::patch(
    std::vector<EntitySummaryEncoding *> &PatchTargets,
    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  for (auto *PatchTarget : PatchTargets) {
    assert(PatchTarget && "EntityLinker::patch: Patch target cannot be null - "
                          "indicates bug in merge logic");
    PatchTarget->patch(EntityResolutionTable);
  }
}

llvm::Error EntityLinker::link(std::unique_ptr<TUSummaryEncoding> Summary) {
  // Check for duplicate TU namespace
  auto [It, Inserted] = ProcessedTUNamespaces.insert(Summary->TUNamespace);
  if (!Inserted) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::DuplicateTUNamespace,
                                Summary->TUNamespace.Name)
        .build();
  }

  TUSummaryEncoding &SummaryRef = *Summary.get();
  auto EntityResolutionTable = resolve(SummaryRef);
  auto PatchTargets = merge(SummaryRef, EntityResolutionTable);
  patch(PatchTargets, EntityResolutionTable);

  return llvm::Error::success();
}
