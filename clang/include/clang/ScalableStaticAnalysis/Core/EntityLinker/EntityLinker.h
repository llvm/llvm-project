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
//  TU summaries may be supplied individually, bundled in a static library, or
//  bundled in one architecture member of a multi-architecture static library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace clang::ssaf {

class MultiArchStaticLibrary;
class StaticLibrary;
class TUSummaryEncoding;

class EntityLinker {
  LUSummaryEncoding Output;

  // Namespaces of the TU summaries folded in, supplied directly or as members
  // of a library.
  std::set<BuildNamespace> ProcessedTUNamespaces;

public:
  /// Constructs an EntityLinker to link TU summaries into a LU summary.
  ///
  /// \param TargetTriple The target triple of the link unit. Every linked
  ///        input must report the same triple.
  /// \param LUNamespace The namespace identifying this link unit.
  EntityLinker(llvm::Triple TargetTriple, NestedBuildNamespace LUNamespace)
      : Output(std::move(TargetTriple), std::move(LUNamespace)) {}

  /// Links a TU summary into a LU summary.
  ///
  /// Deduplicates entities, patches entity ID references in the entity summary,
  /// and merges them into a single data store.
  ///
  /// \param Summary The TU summary to link. Ownership is transferred.
  /// \returns Error if \p Summary reports a different target triple than this
  ///          link unit, if its TU namespace has already been linked, or if
  ///          patching fails; success otherwise. Corrupted summary data
  ///          (missing linkage information, duplicate entity IDs, etc.)
  ///          triggers a fatal error.
  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  /// Links every member of a static library into the LU summary.
  ///
  /// Members are folded in unconditionally, in an unspecified order, exactly as
  /// if each had been passed as an individual TU summary.
  ///
  /// \param Library The static library to link. Ownership is transferred.
  /// \returns Error if \p Library reports a different target triple than this
  ///          link unit or if any member fails to link, success otherwise.
  llvm::Error link(std::unique_ptr<StaticLibrary> Library);

  /// Links the architecture member matching this link unit into the LU summary.
  ///
  /// Members for other architectures are discarded.
  ///
  /// \param Library The multi-arch static library to link. Ownership is
  ///        transferred.
  /// \returns Error if \p Library has no member whose target triple equals this
  ///          link unit's, or if the selected member fails to link; success
  ///          otherwise.
  llvm::Error link(std::unique_ptr<MultiArchStaticLibrary> Library);

  /// Returns the number of TU summaries folded in so far.
  ///
  /// Counts members expanded from libraries as well as TU summaries linked
  /// directly, so it is not the number of link() calls.
  size_t getLinkedTUCount() const { return ProcessedTUNamespaces.size(); }

  /// Returns the accumulated LU summary.
  ///
  /// \returns LU summary containing all the deduplicated and patched entity
  /// summaries.
  LUSummaryEncoding takeOutput() && { return std::move(Output); }

private:
  /// Checks that an input belongs to this link unit's target.
  ///
  /// \param TargetTriple The triple of the input being linked.
  /// \param InputNamespace The namespace naming that input in the diagnostic.
  /// \returns Error if \p TargetTriple differs from this link unit's, success
  ///          otherwise.
  llvm::Error checkTargetTriple(const llvm::Triple &TargetTriple,
                                const BuildNamespace &InputNamespace) const;

  /// Resolves a TU entity name to an LU entity name and ID.
  ///
  /// \param OldName The entity name in the TU namespace.
  /// \param Linkage The linkage determining namespace resolution strategy.
  /// \returns The resolved LU EntityId.
  EntityId resolveEntity(const EntityName &OldName,
                         const EntityLinkage &Linkage,
                         const NestedBuildNamespace &TUNamespace);

  /// Resolves each TU EntityId to its corresponding LU EntityId.
  ///
  /// \param Summary The TU summary whose entities are being resolved.
  /// \returns A map from TU EntityIds to their corresponding LU EntityIds.
  std::map<EntityId, EntityId> resolve(const TUSummaryEncoding &Summary);

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
  /// \returns Error if patching any encoding fails, success otherwise.
  llvm::Error patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
                    const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
