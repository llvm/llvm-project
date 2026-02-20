//===- EntityLinker.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/EntityLinker/EntityLinker.h"
#include "clang/Analysis/Scalable/EntityLinker/EntitySummaryEncoding.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include <cassert>

using namespace clang::ssaf;

//===----------------------------------------------------------------------===//
// Error Message Constants
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

static constexpr const char *EntityLinkerFatalErrorPrefix =
    "EntityLinker: Corrupted TUSummary or logic bug";

static constexpr const char *EntityAlreadyExistsInLinkageTable =
    "{0} - EntityId '{1}' with LinkageType '{2}' already exists in LUSummary";

static constexpr const char *MissingLinkageInformation =
    "{0} - EntityId '{1}' missing linkage information in TUSummary";

static constexpr const char *DuplicateEntityIdInTUSummary =
    "{0} - Duplicate EntityID '{1}' in EntityResolutionTable";

static constexpr const char *EntityNotFoundInResolutionTable =
    "{0} - EntityId '{1}' not found in EntityResolutionTable";

static constexpr const char *FailedToInsertEntityIntoOutputSummary =
    "{0} - Failed to insert data for EntityId '{1}' with LinkageType '{2}' "
    "against SummaryName '{3}' to LUSummary";

static constexpr const char *DuplicateTUNamespace =
    "failed to link TU summary: duplicate namespace '{0}'";

} // namespace ErrorMessages

static NestedBuildNamespace
resolveNamespace(const NestedBuildNamespace &LUNamespace,
                 const NestedBuildNamespace &EntityNamespace,
                 EntityLinkage::LinkageType Linkage) {
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

  auto [_, Inserted] = Output.LinkageTable.try_emplace(NewId, Linkage);
  if (!Inserted) {
    // Insertion failure for `None` and `Internal` linkage is a fatal error
    // because these entities have unique namespaces and should never collide.
    // `External` linkage entities may collide.
    if (Linkage.getLinkage() == EntityLinkage::LinkageType::None ||
        Linkage.getLinkage() == EntityLinkage::LinkageType::Internal) {
      ErrorBuilder::fatal(ErrorMessages::EntityAlreadyExistsInLinkageTable,
                          ErrorMessages::EntityLinkerFatalErrorPrefix,
                          NewId.Index, toString(Linkage.getLinkage()));
    }
  }

  return NewId;
}

std::map<EntityId, EntityId>
EntityLinker::resolve(const TUSummaryEncoding &Summary) {
  std::map<EntityId, EntityId> EntityResolutionTable;

  Summary.IdTable.forEach([&](const EntityName &OldName, const EntityId OldId) {
    auto Iter = Summary.LinkageTable.find(OldId);
    if (Iter == Summary.LinkageTable.end()) {
      ErrorBuilder::fatal(ErrorMessages::MissingLinkageInformation,
                          ErrorMessages::EntityLinkerFatalErrorPrefix,
                          OldId.Index);
    }

    const EntityLinkage &Linkage = Iter->second;

    EntityId NewId = resolveEntity(OldName, Linkage);

    auto [_, Inserted] = EntityResolutionTable.insert({OldId, NewId});
    if (!Inserted) {
      ErrorBuilder::fatal(ErrorMessages::DuplicateEntityIdInTUSummary,
                          ErrorMessages::EntityLinkerFatalErrorPrefix,
                          OldId.Index);
    }
  });

  return EntityResolutionTable;
}

std::vector<EntitySummaryEncoding *>
EntityLinker::merge(TUSummaryEncoding &Summary,
                    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  std::vector<EntitySummaryEncoding *> PatchTargets;

  for (auto &[SN, DataMap] : Summary.Data) {
    auto &OutputSummaryData = Output.Data[SN];

    for (auto &[OldId, ES] : DataMap) {
      auto Iter = EntityResolutionTable.find(OldId);
      if (Iter == EntityResolutionTable.end()) {
        ErrorBuilder::fatal(ErrorMessages::EntityNotFoundInResolutionTable,
                            ErrorMessages::EntityLinkerFatalErrorPrefix,
                            OldId.Index);
      }

      const auto NewId = Iter->second;

      auto [It, Inserted] = OutputSummaryData.try_emplace(NewId, std::move(ES));

      if (Inserted) {
        PatchTargets.push_back(It->second.get());
      } else {
        // Safe to retrieve linkage using .at since the resolve step ensures
        // linkage information is always present for every OldId.
        auto Linkage = Summary.LinkageTable.at(OldId).getLinkage();

        // Insertion should never fail for `None` and `Internal` linkage
        // entities because these entities will have different namespaces across
        // TUs even if their names match.
        if (Linkage == EntityLinkage::LinkageType::None ||
            Linkage == EntityLinkage::LinkageType::Internal) {
          ErrorBuilder::fatal(
              ErrorMessages::FailedToInsertEntityIntoOutputSummary,
              ErrorMessages::EntityLinkerFatalErrorPrefix, NewId.Index,
              toString(Linkage), SN.str());
        }

        // Insertion is expected to fail for duplicate occurrences of `External`
        // linkage entities. TODO - report these cases in a "debug" mode to help
        // debug potential ODR violations.
      }
    }
  }

  return PatchTargets;
}

void EntityLinker::patch(
    const std::vector<EntitySummaryEncoding *> &PatchTargets,
    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  for (auto *PatchTarget : PatchTargets) {
    assert(PatchTarget && "EntityLinker::patch: Patch target cannot be null");
    PatchTarget->patch(EntityResolutionTable);
  }
}

llvm::Error EntityLinker::link(std::unique_ptr<TUSummaryEncoding> Summary) {
  auto [_, Inserted] = ProcessedTUNamespaces.insert(Summary->TUNamespace);
  if (!Inserted) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::DuplicateTUNamespace,
                                Summary->TUNamespace.Name)
        .build();
  }

  TUSummaryEncoding &SummaryRef = *Summary;

  auto EntityResolutionTable = resolve(SummaryRef);
  auto PatchTargets = merge(SummaryRef, EntityResolutionTable);
  patch(PatchTargets, EntityResolutionTable);

  return llvm::Error::success();
}
