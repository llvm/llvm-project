//===- EntityLinker.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/EntitySummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/FormatProviders.h"
#include <cassert>

using namespace clang::ssaf;

//===----------------------------------------------------------------------===//
// Error Message Constants
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

static constexpr const char *EntityLinkerFatalErrorPrefix =
    "EntityLinker: Corrupted TUSummary or logic bug";

static constexpr const char *EntityAlreadyExistsInLinkageTable =
    "{0} - {1} with {2} already exists in LUSummary";

static constexpr const char *MissingLinkageInformation =
    "{0} - {1} missing linkage information in TUSummary";

static constexpr const char *DuplicateEntityIdInTUSummary =
    "{0} - Duplicate {1} in EntityResolutionTable";

static constexpr const char *EntityNotFoundInResolutionTable =
    "{0} - {1} not found in EntityResolutionTable";

static constexpr const char *FailedToInsertEntityIntoOutputSummary =
    "{0} - Failed to insert data for {1} with {2} against {3} to LUSummary";

static constexpr const char *DuplicateTUNamespace =
    "failed to link TU summary: duplicate {0}";

} // namespace ErrorMessages

static NestedBuildNamespace
resolveNamespace(const NestedBuildNamespace &LUNamespace,
                 const NestedBuildNamespace &TUNamespace,
                 const NestedBuildNamespace &EntityNamespace,
                 EntityLinkageType Linkage) {
  switch (Linkage) {
  case EntityLinkageType::None:
  case EntityLinkageType::Internal:
    // Qualify with the TU namespace first (to disambiguate across TUs),
    // then with the LU namespace.
    return EntityNamespace.makeQualified(TUNamespace)
        .makeQualified(LUNamespace);
  case EntityLinkageType::External:
    return NestedBuildNamespace(LUNamespace);
  }

  llvm_unreachable("Unhandled EntityLinkageType variant");
}

EntityId EntityLinker::resolveEntity(const EntityName &OldName,
                                     const EntityLinkage &Linkage,
                                     const NestedBuildNamespace &TUNamespace) {
  NestedBuildNamespace NewNamespace = resolveNamespace(
      Output.LUNamespace, TUNamespace, OldName.Namespace, Linkage.getLinkage());

  EntityName NewName(OldName.USR, OldName.Suffix, NewNamespace);

  // NewId construction will always return a fresh id for `None` and `Internal`
  // linkage entities since their namespaces will be different even if their
  // names clash. For `External` linkage entities with identical names this
  // function will return the id assigned at the first insertion.
  EntityId NewId = Output.IdTable.getId(NewName);

  auto [_, Inserted] = Output.LinkageTable.try_emplace(NewId, Linkage);
  if (!Inserted) {
    // Insertion failure for `None` and `Internal` linkage is a fatal error
    // because these entities have unique namespaces and should never collide.
    // `External` linkage entities may collide.
    if (Linkage.getLinkage() == EntityLinkageType::None ||
        Linkage.getLinkage() == EntityLinkageType::Internal) {
      ErrorBuilder::fatal(ErrorMessages::EntityAlreadyExistsInLinkageTable,
                          ErrorMessages::EntityLinkerFatalErrorPrefix, NewId,
                          Linkage);
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
                          ErrorMessages::EntityLinkerFatalErrorPrefix, OldId);
    }

    const EntityLinkage &Linkage = Iter->second;

    EntityId NewId = resolveEntity(OldName, Linkage,
                                   NestedBuildNamespace(Summary.TUNamespace));

    auto [_, Inserted] = EntityResolutionTable.insert({OldId, NewId});
    if (!Inserted) {
      ErrorBuilder::fatal(ErrorMessages::DuplicateEntityIdInTUSummary,
                          ErrorMessages::EntityLinkerFatalErrorPrefix, OldId);
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
                            ErrorMessages::EntityLinkerFatalErrorPrefix, OldId);
      }

      const auto NewId = Iter->second;

      auto [It, Inserted] = OutputSummaryData.try_emplace(NewId, std::move(ES));

      if (Inserted) {
        PatchTargets.push_back(It->second.get());
      } else {
        // Safe to retrieve linkage using .at since the resolve step ensures
        // linkage information is always present for every OldId.
        auto Linkage = Summary.LinkageTable.at(OldId);

        // Insertion should never fail for `None` and `Internal` linkage
        // entities because these entities will have different namespaces across
        // TUs even if their names match.
        if (Linkage.getLinkage() == EntityLinkageType::None ||
            Linkage.getLinkage() == EntityLinkageType::Internal) {
          ErrorBuilder::fatal(
              ErrorMessages::FailedToInsertEntityIntoOutputSummary,
              ErrorMessages::EntityLinkerFatalErrorPrefix, NewId, Linkage, SN);
        }

        // TODO: Insertion is expected to fail for duplicate occurrences of
        // `External` linkage entities. Report these cases in a "debug" mode to
        // help debug potential ODR violations.
      }
    }
  }

  return PatchTargets;
}

llvm::Error
EntityLinker::patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
                    const std::map<EntityId, EntityId> &EntityResolutionTable) {
  for (auto *PatchTarget : PatchTargets) {
    assert(PatchTarget && "EntityLinker::patch: Patch target cannot be null");

    if (auto Err = PatchTarget->patch(EntityResolutionTable)) {
      return Err;
    }
  }
  return llvm::Error::success();
}

llvm::Error EntityLinker::link(std::unique_ptr<TUSummaryEncoding> Summary) {
  auto [_, Inserted] = ProcessedTUNamespaces.insert(Summary->TUNamespace);
  if (!Inserted) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::DuplicateTUNamespace,
                                Summary->TUNamespace)
        .build();
  }

  TUSummaryEncoding &SummaryRef = *Summary;

  auto EntityResolutionTable = resolve(SummaryRef);
  auto PatchTargets = merge(SummaryRef, EntityResolutionTable);
  return patch(PatchTargets, EntityResolutionTable);
}
