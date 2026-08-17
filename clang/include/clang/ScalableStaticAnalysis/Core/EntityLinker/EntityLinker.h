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

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LinkageRules.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace clang::ssaf {

class TUSummaryEncoding;

/// How the linker reports entities that no linked TU defines.
enum class UnresolvedPolicy : uint8_t {
  Ignore, ///< do not check
  Warn,   ///< report on stderr and keep linking
  Error   ///< fail the link
};

/// How the linker reports summary data that disagrees between two definitions
/// required to be identical.
enum class ODRMismatchPolicy : uint8_t {
  Ignore, ///< do not check
  Warn,   ///< report on stderr and keep linking
  Error   ///< fail the link
};

class EntityLinker {
  friend class TestFixture;

  /// Records, for each LU EntityId a TU resolved to, whether that TU's summary
  /// data for the entity is the copy to keep. It is false when an earlier TU's
  /// occurrence won the linkage reconciliation, in which case the data the TU
  /// carries for that entity is dropped.
  using DataSelectionMap = std::map<EntityId, bool>;

  LUSummaryEncoding Output;
  std::set<BuildNamespace> ProcessedTUNamespaces;

  /// The resolution rules for the link unit's object format. Owned statically
  /// by LinkageRules::forTarget().
  const LinkageRules *Rules;

  /// Whether a multiple definition is reported as a warning rather than
  /// failing the link.
  bool WarnOnMultipleDefinitions;

  /// How entities that no linked TU defines are reported.
  UnresolvedPolicy UnresolvedSymbols;

  /// How summary data that disagrees between ODR definitions is reported.
  ODRMismatchPolicy ODRMismatch;

  /// Summary data displaced by a duplicate ODR definition, kept only when
  /// \c ODRMismatch asks for the comparison. Per-link scratch, compared once
  /// the whole TU has been patched into the LU EntityId space.
  struct DisplacedData {
    SummaryName Name;                            ///< the summary it belongs to
    EntityId Id;                                 ///< the LU entity it describes
    std::unique_ptr<EntitySummaryEncoding> Data; ///< the displaced encoding
  };
  std::vector<DisplacedData> Displaced;

  /// Names of the LU entities, kept so that finalization can name them in
  /// diagnostics. EntityIdTable offers no reverse lookup.
  std::map<EntityId, EntityName> EntityNames;

public:
  /// Constructs an EntityLinker to link TU summaries into a LU summary.
  ///
  /// \param TargetTriple The target triple of the link unit. It selects the
  ///        resolution rules to emulate, and every linked TU must report a
  ///        compatible triple.
  /// \param LUNamespace The namespace identifying this link unit.
  /// \param WarnOnMultipleDefinitions Report an external entity defined by more
  ///        than one TU as a warning on stderr and keep linking. By default
  ///        such an entity fails the link, matching a linker's
  ///        multiple-definition rule.
  /// \param UnresolvedSymbols How to report entities that no linked TU defines.
  ///        Defaults to \c Ignore because a link unit is usually an
  ///        intermediate artifact, where unresolved references are expected.
  /// \param ODRMismatch How to report two definitions that were required to be
  ///        identical but whose summary data differs. Defaults to \c Ignore
  ///        because the check costs an extra patch of every displaced encoding.
  EntityLinker(llvm::Triple TargetTriple, NestedBuildNamespace LUNamespace,
               bool WarnOnMultipleDefinitions = false,
               UnresolvedPolicy UnresolvedSymbols = UnresolvedPolicy::Ignore,
               ODRMismatchPolicy ODRMismatch = ODRMismatchPolicy::Ignore)
      : Output(std::move(TargetTriple), std::move(LUNamespace)),
        Rules(&LinkageRules::forTarget(Output.TargetTriple)),
        WarnOnMultipleDefinitions(WarnOnMultipleDefinitions),
        UnresolvedSymbols(UnresolvedSymbols), ODRMismatch(ODRMismatch) {}

  /// Links a TU summary into a LU summary.
  ///
  /// Deduplicates entities, patches entity ID references in the entity summary,
  /// and merges them into a single data store.
  ///
  /// \param Summary The TU summary to link. Ownership is transferred.
  /// \returns Error if the TU namespace has already been linked or if patching
  ///          fails, success otherwise. Corrupted summary data (missing linkage
  ///          information, duplicate entity IDs, etc.) triggers a fatal error,
  ///          as does an external entity defined by more than one TU unless
  ///          warnings were requested.
  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  /// Returns the accumulated LU summary.
  ///
  /// Runs finalization first: rules that need the complete link unit, such as
  /// reporting entities no TU defined and demoting hidden entities, cannot run
  /// per-TU because a later TU can change the answer.
  ///
  /// \returns LU summary containing all the deduplicated and patched entity
  /// summaries.
  LUSummaryEncoding takeOutput() && {
    finalize();
    return std::move(Output);
  }

private:
  /// Applies the rules that need the whole link unit. See takeOutput().
  void finalize();

  /// Reports entities that no linked TU defined, under \c UnresolvedSymbols.
  ///
  /// An undefined weak reference is legal by design and is never reported:
  ///
  ///   __attribute__((weak)) int maybe_missing(void);
  ///   $ ld.lld -o out undef_weak_main.o     -> links
  ///
  /// whereas a strong one fails when linking an executable. We default to
  /// Ignore rather than emulating a specific platform because a link unit is
  /// typically intermediate; see the plan's §3.1.
  void reportUnresolvedEntities() const;

  /// Demotes entities the target treats as hidden from `External` to
  /// `Internal`, so a later link stage cannot resolve against them.
  ///
  /// This is the link-unit analogue of a hidden ELF symbol becoming LOCAL and
  /// dropping out of .dynsym:
  ///
  ///   // tu1.c
  ///   int v(void) { return 1; }
  ///   // tu2.c
  ///   __attribute__((visibility("hidden"))) int v(void);
  ///   $ ld.lld -shared -o out.so tu1.o tu2.o
  ///     v: FUNC LOCAL HIDDEN, absent from .dynsym
  ///
  /// Only the linkage type changes: the entity keeps its name, because every
  /// EntityId already patched into the summary data refers to it, and external
  /// names are LU-qualified so they cannot collide across link units.
  /// (ELF doc §4 V1.)
  void demoteHiddenEntities();

  /// The binding the target emits for \p Linkage, per the active rules.
  EntityBinding effectiveBinding(const EntityLinkage &Linkage) const;

  /// Compares each displaced encoding against the one kept in its place.
  ///
  /// Only meaningful for entities whose definitions were required to be
  /// identical: for a plain weak definition, differing bodies are legal and
  /// the same check would be noise. Runs after patch(), because the two
  /// encodings are only comparable once both are in the LU EntityId space.
  ///
  /// \param TUNamespace The namespace of the TU whose data was displaced.
  void reportODRMismatches(const NestedBuildNamespace &TUNamespace);

  /// Returns true if \p Current and \p Incoming are two definitions of the
  /// same external entity that the target cannot both accept.
  ///
  /// \param Current The linkage already recorded for the entity.
  /// \param Incoming The linkage of the occurrence being linked.
  bool isConflictingDefinition(const EntityLinkage &Current,
                               const EntityLinkage &Incoming) const;

  /// Decides whether the summary data of the occurrence being linked replaces
  /// the data already linked for the same external entity.
  ///
  /// A definition beats a declaration, and among definitions the stronger
  /// binding wins. Every other case keeps the data already linked, so ties
  /// resolve in favour of whichever occurrence was linked first. This is the
  /// one reconciliation rule that is deliberately order-sensitive, and every
  /// platform agrees on it:
  ///
  ///   __attribute__((weak)) int f(void) { return 1; }   // weak_a.c
  ///   __attribute__((weak)) int f(void) { return 2; }   // weak_b.c
  ///   $ ld.lld -o out main.o weak_a.o weak_b.o
  ///     f -> returns 1, the first one linked
  ///
  /// (ELF doc §1 P4, Mach-O doc §1 M4.)
  ///
  /// \param Current The linkage already recorded for the entity.
  /// \param Incoming The linkage of the occurrence being linked.
  bool incomingDataWins(const EntityLinkage &Current,
                        const EntityLinkage &Incoming) const;

  /// Reconciles two occurrences of the same external entity into the linkage
  /// recorded for it.
  ///
  /// Binding and coalescing describe definitions, so a declaration contributes
  /// neither when the other occurrence defines the entity — every platform
  /// agrees that a weak declaration leaves no trace on a common definition
  /// (ELF doc §6.1, Mach-O doc §6.1, COFF doc §6.2). Visibility merges across
  /// all occurrences, including ones that lose resolution.
  ///
  /// The result is commutative except where the target itself is
  /// order-dependent; see LinkageRules::isOrderDependentMerge().
  ///
  /// \param Current The linkage already recorded for the entity.
  /// \param Incoming The linkage of the occurrence being linked.
  /// \pre Both occurrences have the same linkage type. Namespace resolution
  ///      guarantees this for every collision the linker reconciles.
  EntityLinkage mergeLinkage(const EntityLinkage &Current,
                             const EntityLinkage &Incoming) const;

  /// Records \p TUNamespace as linked into this link unit.
  ///
  /// \param TUNamespace The namespace of the TU being linked.
  /// \returns Error if a TU with this namespace has already been linked,
  ///          success otherwise.
  llvm::Error checkTUNotAlreadyLinked(const BuildNamespace &TUNamespace);

  /// Verifies that \p TUTriple selects the same resolution rules as the link
  /// unit's own triple.
  ///
  /// Rules differ enough between formats that applying the wrong ones is
  /// silently incorrect: Mach-O ranks a weak definition above a common while
  /// ELF and COFF do the reverse, and COFF rejects two weak definitions the
  /// others accept. The OS version is ignored, since it does not affect
  /// resolution.
  ///
  /// \returns Error if the TU targets a different platform, success otherwise.
  llvm::Error checkTUTargetMatches(const llvm::Triple &TUTriple,
                                   const BuildNamespace &TUNamespace) const;

  /// Fails the link if \p Linkage holds a value the target cannot express.
  ///
  /// \param Name The entity the linkage belongs to, for diagnostics.
  void reportIfNotRepresentable(const EntityLinkage &Linkage,
                                const EntityName &Name) const;

  /// Warns when reconciling these two occurrences gives an order-dependent
  /// result, which means the program itself is ambiguous.
  void reportIfOrderDependent(const EntityLinkage &Current,
                              const EntityLinkage &Incoming,
                              const EntityName &Name) const;

  /// Fails the link if a collision happened on a non-`External` entity.
  ///
  /// `None` and `Internal` entities are qualified with their TU namespace, so
  /// they can never collide across TUs. A collision means the TU summary is
  /// corrupted or namespace resolution is buggy.
  ///
  /// \param NewId The LU EntityId that collided.
  /// \param Linkage The linkage of the colliding occurrence.
  static void reportIfLinkageIsNotExternal(EntityId NewId,
                                           const EntityLinkage &Linkage);

  /// Reports an external entity that more than one TU defines.
  ///
  /// Does nothing unless \p Current and \p Incoming both strongly define the
  /// entity. Otherwise it terminates with a fatal error, or emits a warning on
  /// stderr and returns if \c WarnOnMultipleDefinitions is set.
  ///
  /// \param Current The linkage already recorded for the entity.
  /// \param Incoming The linkage of the occurrence being linked.
  /// \param Name The LU entity name the two occurrences resolved to.
  /// \param TUNamespace The namespace of the TU carrying the extra definition.
  /// \returns True if the two occurrences conflict, in which case the caller
  ///          keeps the definition already linked and ignores \p Incoming
  ///          entirely rather than reconciling the two.
  bool reportIfDefinitionsConflict(
      const EntityLinkage &Current, const EntityLinkage &Incoming,
      const EntityName &Name, const NestedBuildNamespace &TUNamespace) const;

  /// Resolves a TU entity name to an LU entity name and ID.
  ///
  /// \param OldName The entity name in the TU namespace.
  /// \param Linkage The linkage determining namespace resolution strategy.
  /// \param TUNamespace The namespace of the TU being linked.
  /// \returns The resolved LU EntityId, paired with whether this TU's summary
  ///          data for it is the copy to keep.
  std::pair<EntityId, bool>
  resolveEntity(const EntityName &OldName, const EntityLinkage &Linkage,
                const NestedBuildNamespace &TUNamespace);

  /// Resolves each TU EntityId to its corresponding LU EntityId.
  ///
  /// \param Summary The TU summary whose entities are being resolved.
  /// \returns A map from TU EntityIds to their corresponding LU EntityIds,
  ///          paired with the data selection for each LU EntityId resolved.
  std::pair<EntityResolutionMap, DataSelectionMap>
  resolve(const TUSummaryEncoding &Summary);

  /// Merges all summary data from a TU summary into the LU Summary.
  ///
  /// \param Summary The TU summary whose data is being merged.
  /// \param Resolution Map from TU EntityIds to LU EntityIds.
  /// \param DataSelection Which LU EntityIds this TU owns the data for.
  /// \returns Pointers to each EntitySummaryEncoding successfully merged.
  std::vector<EntitySummaryEncoding *>
  merge(TUSummaryEncoding &Summary, const EntityResolutionMap &Resolution,
        const DataSelectionMap &DataSelection);

  /// Merges one summary's data from a TU summary into the LU summary.
  ///
  /// \param SN The summary the data belongs to.
  /// \param DataMap The TU's encodings for \p SN, keyed by TU EntityId.
  /// \param Summary The TU summary being merged.
  /// \param Resolution Map from TU EntityIds to LU EntityIds.
  /// \param DataSelection Which LU EntityIds this TU owns the data for.
  /// \returns Pointers to each EntitySummaryEncoding successfully merged.
  std::vector<EntitySummaryEncoding *>
  mergeSummaryData(const SummaryName &SN, EntityDataMap &DataMap,
                   const TUSummaryEncoding &Summary,
                   const EntityResolutionMap &Resolution,
                   const DataSelectionMap &DataSelection);

  /// Merges one entity's encoding into the LU summary data for a summary.
  ///
  /// The encoding is dropped when an earlier TU already contributed the copy
  /// to keep for \p NewId.
  ///
  /// \param OutputData The LU summary data to merge into.
  /// \param SN The summary the data belongs to, for diagnostics.
  /// \param NewId The LU EntityId the data describes.
  /// \param Encoding The TU's encoding for \p NewId. Ownership is transferred
  ///        only when the encoding is kept.
  /// \param Linkage The TU's linkage for the entity.
  /// \param DataFromIncoming Whether \p Encoding is the copy to keep.
  /// \returns The merged encoding to patch, or nullptr if it was dropped
  ///          without being kept for comparison.
  EntitySummaryEncoding *
  mergeEntityData(EntityDataMap &OutputData, const SummaryName &SN,
                  EntityId NewId,
                  std::unique_ptr<EntitySummaryEncoding> &Encoding,
                  const EntityLinkage &Linkage, bool DataFromIncoming);

  /// Patches EntityId references in merged summary data.
  ///
  /// \param PatchTargets Vector of summary encodings that need patching.
  /// \param Resolution Map from TU EntityIds to LU EntityIds.
  /// \returns Error if patching any encoding fails, success otherwise.
  llvm::Error patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
                    const EntityResolutionMap &Resolution);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_ENTITYLINKER_H
