//===- EntityLinker.h - Class for linking entities --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the EntityLinker class that combines multiple TU summaries
//  into a unified LU summary by deduplicating entities and patching summaries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H

#include "clang/Analysis/Scalable/EntityLinker/LUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace clang::ssaf {

class EntityLinkage;
class EntityName;
class EntitySummaryEncoding;
class TUSummaryEncoding;

class EntityLinker {
  LUSummaryEncoding Output;
  std::set<BuildNamespace> ProcessedTUNamespaces;

public:
  /// Constructs an EntityLinker for a link unit.
  ///
  /// \param LUNamespace The namespace identifying this link unit.
  EntityLinker(NestedBuildNamespace LUNamespace)
      : Output(std::move(LUNamespace)) {}

  /// Links a translation unit summary into the link unit summary.
  ///
  /// Processes entity names, resolves namespace conflicts based on linkage,
  /// deduplicates entities, and patches entity ID references in the summary
  /// data. The provided TU summary is consumed by this operation.
  ///
  /// \param Summary The TU summary to link. Ownership is transferred.
  /// \returns Error if the TU namespace has already been linked, success
  ///          otherwise. Corrupted summary data (missing linkage information,
  ///          duplicate entity IDs) triggers a fatal error.
  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  /// Returns the accumulated link unit summary.
  ///
  /// \returns A const reference to the linked output containing all
  ///          deduplicated and patched entity summaries.
  const LUSummaryEncoding &getOutput() const { return Output; }

  /// Returns the accumulated link unit summary.
  ///
  /// \returns A mutable reference to the linked output.
  LUSummaryEncoding &getOutput() { return Output; }

private:
  /// Resolves a TU entity name to an LU entity name and ID.
  ///
  /// Determines the appropriate namespace for the entity based on its linkage
  /// type. Entities with None or Internal linkage are scoped to their TU,
  /// while External linkage entities are scoped to the LU. Creates or retrieves
  /// the corresponding EntityId in the output LinkageTable.
  ///
  /// For None and Internal linkage entities, duplicate insertion in the
  /// LinkageTable triggers a fatal error (indicates corrupted data).
  /// For External linkage entities, duplicate insertion is allowed (expected
  /// for multiple definitions of the same entity).
  ///
  /// \param OldName The entity name in the TU namespace.
  /// \param Linkage The linkage type determining namespace resolution strategy.
  /// \returns The resolved LU EntityId.
  EntityId resolveEntity(const EntityName &OldName,
                         const EntityLinkage &Linkage);

  /// Builds a map from each TU EntityId to its corresponding LU EntityId.
  ///
  /// Iterates over all entities in Summary's IdTable, looks up their linkage,
  /// and calls resolveEntity() to obtain the LU-scoped EntityId. The resulting
  /// map is used by merge() and patch() to translate TU IDs into LU IDs.
  ///
  /// Corrupted input triggers a fatal error: missing linkage for an entity in
  /// IdTable, or a duplicate EntityId appearing under two different names.
  ///
  /// \param Summary The TU summary whose entities are being resolved.
  /// \returns A map from TU EntityIds to their corresponding LU EntityIds.
  std::map<EntityId, EntityId> resolve(TUSummaryEncoding &Summary);

  /// Merges all summary data from a TU into the LU output.
  ///
  /// Iterates over every (SummaryName, EntityId, data) entry in Summary.
  /// Each TU EntityId is translated to its LU EntityId via
  /// EntityResolutionTable, then the data is moved into the corresponding
  /// output map entry.
  ///
  /// For External linkage entities, a duplicate entry (same LU EntityId already
  /// present for a given SummaryName) is silently dropped — first occurrence
  /// wins. For None and Internal linkage entities, a duplicate entry indicates
  /// corrupted data and triggers a fatal error.
  ///
  /// \param Summary The TU summary whose data is being merged (data is moved
  ///        out).
  /// \param EntityResolutionTable Map from TU EntityIds to LU EntityIds,
  ///        as produced by resolve().
  /// \returns Pointers to each EntitySummaryEncoding successfully inserted into
  ///          the output, which must subsequently be patched via patch().
  std::vector<EntitySummaryEncoding *>
  merge(TUSummaryEncoding &Summary,
        std::map<EntityId, EntityId> EntityResolutionTable);

  /// Patches EntityId references in merged summary data.
  ///
  /// Calls the patch() method on each EntitySummaryEncoding that was
  /// successfully merged into the LU output, updating all embedded EntityId
  /// references from TU IDs to LU IDs using the provided resolution table.
  ///
  /// \param PatchTargets Vector of summary encodings that need patching.
  /// \param EntityResolutionTable Map from TU EntityIds to LU EntityIds.
  void patch(std::vector<EntitySummaryEncoding *> &PatchTargets,
             const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // end namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
