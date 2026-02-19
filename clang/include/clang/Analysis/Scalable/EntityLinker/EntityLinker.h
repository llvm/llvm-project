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
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace clang::ssaf {

class BuildNamespace;
class EntityId;
class EntityLinkage;
class EntityName;
class EntitySummaryEncoding;
class TUSummaryEncoding;

class EntityLinker {
  LUSummaryEncoding Output;
  std::set<BuildNamespace> ProcessedTUNamespaces;

public:
  /// Constructs an EntityLinker to link TU summaries into a LU summary.
  ///
  /// \param LUNamespace The namespace identifying this link unit.
  EntityLinker(NestedBuildNamespace LUNamespace)
      : Output(std::move(LUNamespace)) {}

  /// Links a TU summary into a LU summary.
  ///
  /// Deduplicates entities, patches entity ID references in the entity summary,
  /// and merges them into a single data store. The provided TU summary is
  /// consumed by this operation.
  ///
  /// \param Summary The TU summary to link. Ownership is transferred.
  /// \returns Error if the TU namespace has already been linked, success
  ///          otherwise. Corrupted summary data (missing linkage information,
  ///          duplicate entity IDs, etc.) triggers a fatal error.
  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  /// Returns the accumulated LU summary.
  ///
  /// \returns LU summary containing all the deduplicated and patched entity
  /// summaries.
  const LUSummaryEncoding &getOutput() const { return Output; }

private:
  /// Resolves a TU entity name to an LU entity name and ID.
  ///
  /// \param OldName The entity name in the TU namespace.
  /// \param Linkage The linkage determining namespace resolution strategy.
  /// \returns The resolved LU EntityId.
  EntityId resolveEntity(const EntityName &OldName,
                         const EntityLinkage &Linkage);

  /// Resolves each TU EntityId to its corresponding LU EntityId.
  ///
  /// \param Summary The TU summary whose entities are being resolved.
  /// \returns A map from TU EntityIds to their corresponding LU EntityIds.
  std::map<EntityId, EntityId> resolve(TUSummaryEncoding &Summary);

  /// Merges all summary data from a TU summary into the LU Summary.
  ///
  /// \param Summary The TU summary whose data is being merged.
  /// \param EntityResolutionTable Map from TU EntityIds to LU EntityIds.
  /// \returns Pointers to each EntitySummaryEncoding successfully merged.
  std::vector<EntitySummaryEncoding *>
  merge(TUSummaryEncoding &Summary,
        const std::map<EntityId, EntityId> &EntityResolutionTable);

  /// Patches EntityId references in merged summary data.
  ///
  /// \param PatchTargets Vector of summary encodings that need patching.
  /// \param EntityResolutionTable Map from TU EntityIds to LU EntityIds.
  void patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
             const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
