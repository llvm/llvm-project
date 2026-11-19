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
//  Shared libraries, supplied individually or as one architecture member of a
//  multi-architecture shared library are folded in the same way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace clang::ssaf {

class MultiArchSharedLibrary;
class MultiArchStaticLibrary;
class StaticLibrary;
class TUSummaryEncoding;

/// Computes the build namespace an entity should carry after linking, given its
/// linkage.
NestedBuildNamespace
resolveNamespace(const NestedBuildNamespace &LUNamespace,
                 const NestedBuildNamespace &TUNamespace,
                 const NestedBuildNamespace &EntityNamespace,
                 EntityLinkageType Linkage);

class EntityLinker {
  LUSummaryEncoding Output;

  // Namespaces of the TU summaries folded in, supplied directly or as members
  // of a library.
  std::set<BuildNamespace> ProcessedTUNamespaces;

  // Namespaces of the shared libraries folded in, supplied directly or as
  // members of a multi-arch shared library.
  std::set<NestedBuildNamespace> ProcessedLUNamespaces;

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

  /// Links a shared library into the LU summary.
  ///
  /// The shared library is an already linked LU summary. Its entities are
  /// resolved against this link unit exactly as a TU summary's are, except
  /// that entities without external linkage are already qualified by the
  /// shared library's namespace and so are not qualified by a TU namespace.
  ///
  /// \param Library The shared library to link. Ownership is transferred.
  /// \returns Error if \p Library reports a different target triple than this
  ///          link unit, if its namespace is this link unit's or has already
  ///          been linked, or if patching fails; success otherwise.
  llvm::Error link(std::unique_ptr<LUSummaryEncoding> Library);

  /// Links the architecture member matching this link unit into the LU summary.
  ///
  /// Members for other architectures are discarded.
  ///
  /// \param Library The multi-arch shared library to link. Ownership is
  ///        transferred.
  /// \returns Error if \p Library has no member whose target triple equals this
  ///          link unit's, or if the selected member fails to link; success
  ///          otherwise.
  llvm::Error link(std::unique_ptr<MultiArchSharedLibrary> Library);

  /// Returns the number of TU summaries folded in so far.
  ///
  /// Counts members expanded from static libraries as well as TU summaries
  /// linked directly, so it is not the number of link() calls. TUs already
  /// linked into a shared library are not counted.
  size_t getLinkedTUCount() const { return ProcessedTUNamespaces.size(); }

  /// Returns the number of shared libraries folded in so far.
  ///
  /// Counts members selected from multi-arch shared libraries as well as
  /// shared libraries linked directly.
  size_t getLinkedSharedLibraryCount() const {
    return ProcessedLUNamespaces.size();
  }

  /// Returns the accumulated LU summary.
  ///
  /// \returns LU summary containing all the deduplicated and patched entity
  /// summaries.
  LUSummaryEncoding takeOutput() && { return std::move(Output); }

private:
  /// Checks that an input belongs to this link unit's target.
  ///
  /// \param TargetTriple The triple of the input being linked.
  /// \param InputNamespace The namespace naming that input in the diagnostic, a
  ///        BuildNamespace or a NestedBuildNamespace.
  /// \returns Error if \p TargetTriple differs from this link unit's, success
  ///          otherwise.
  template <typename NamespaceT>
  llvm::Error checkTargetTriple(const llvm::Triple &TargetTriple,
                                const NamespaceT &InputNamespace) const;

  /// Links the architecture member of a multi-arch library matching this link
  /// unit, discarding the others.
  ///
  /// \param Library A MultiArchStaticLibrary or MultiArchSharedLibrary.
  /// \param KindName The library kind named in the diagnostic, e.g. "static
  ///        library".
  /// \returns Error if \p Library has no member whose target triple equals this
  ///          link unit's, or if the selected member fails to link; success
  ///          otherwise.
  template <typename MultiArchLibraryT>
  llvm::Error linkMatchingMember(std::unique_ptr<MultiArchLibraryT> Library,
                                 llvm::StringRef KindName);

  /// Resolves a TU entity name to an LU entity name and ID.
  ///
  /// \param OldName The entity name in the input's namespace.
  /// \param Linkage The linkage determining namespace resolution strategy.
  /// \param InputNamespace The namespace qualifying entities without external
  ///        linkage before this link unit's namespace does.
  /// \returns The resolved LU EntityId.
  EntityId resolveEntity(const EntityName &OldName,
                         const EntityLinkage &Linkage,
                         const NestedBuildNamespace &InputNamespace);

  /// Resolves each input EntityId to its corresponding LU EntityId.
  ///
  /// \param IdTable The entities of the input being resolved.
  /// \param LinkageTable The linkage of each entity in \p IdTable.
  /// \param InputNamespace The namespace qualifying entities without external
  ///        linkage before this link unit's namespace does: the TU namespace
  ///        for a TU summary and empty for a shared library, whose entities
  ///        are already qualified.
  /// \returns A map from input EntityIds to their corresponding LU EntityIds.
  std::map<EntityId, EntityId>
  resolve(const EntityIdTable &IdTable,
          const std::map<EntityId, EntityLinkage> &LinkageTable,
          const NestedBuildNamespace &InputNamespace);

  /// Merges all summary data from an input into the LU Summary.
  ///
  /// \param Data The summary data being merged. Merged encodings are moved out.
  /// \param LinkageTable The linkage of each entity in \p Data.
  /// \param EntityResolutionTable Map from input EntityIds to LU EntityIds.
  /// \returns Pointers to each EntitySummaryEncoding successfully merged.
  std::vector<EntitySummaryEncoding *>
  merge(std::map<SummaryName,
                 std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
            &Data,
        const std::map<EntityId, EntityLinkage> &LinkageTable,
        const std::map<EntityId, EntityId> &EntityResolutionTable);

  /// Patches EntityId references in merged summary data.
  ///
  /// \param PatchTargets Vector of summary encodings that need patching.
  /// \param EntityResolutionTable Map from input EntityIds to LU EntityIds.
  /// \returns Error if patching any encoding fails, success otherwise.
  llvm::Error patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
                    const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
