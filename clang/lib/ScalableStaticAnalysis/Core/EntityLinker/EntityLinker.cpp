//===- EntityLinker.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntitySummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

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

static constexpr const char *MultipleEntitiesResolveToSameId =
    "{0} - Multiple entities in the TU summary resolve to {1}";

static constexpr const char *FailedToInsertEntityIntoOutputSummary =
    "{0} - Failed to insert data for {1} with {2} against {3} to LUSummary";

static constexpr const char *DuplicateTUNamespace =
    "failed to link TU summary: duplicate {0}";

static constexpr const char *MultipleDefinition =
    "multiple definition of {0} in {1}";

static constexpr const char *TargetTripleMismatch =
    "failed to link TU summary: {0} targets '{1}', which resolves symbols as "
    "{2}, but the link unit targets '{3}', which resolves them as {4}";

static constexpr const char *UnrepresentableLinkage =
    "{0} has {1}, which a {2} target cannot represent";

static constexpr const char *OrderDependentMerge =
    "visibility of common symbol {0} differs between translation units; the "
    "linked result depends on link order";

static constexpr const char *UnresolvedEntity = "undefined symbol: {0}";

static constexpr const char *ODRMismatch =
    "{0} summary data for {1} differs between {2} and an earlier translation "
    "unit, but its definitions are required to be identical";

} // namespace ErrorMessages

//===----------------------------------------------------------------------===//
// Fatal Error Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Reports a corrupted TU summary or logic bug and terminates execution.
///
/// Supplies \c EntityLinkerFatalErrorPrefix as the leading `{0}` of every
/// message in \c ErrorMessages, so callers pass only the remaining arguments.
template <typename... Args>
[[noreturn]] void fatal(const char *Fmt, Args &&...ArgVals) {
  ErrorBuilder::fatal(Fmt, ErrorMessages::EntityLinkerFatalErrorPrefix,
                      std::forward<Args>(ArgVals)...);
}

/// Looks \p Key up in \p Map, reporting a fatal error if it is absent.
///
/// \returns A reference to the mapped value, which is owned by \p Map.
template <typename MapT, typename... Args>
const typename MapT::mapped_type &
lookupOrFatal(const MapT &Map, const typename MapT::key_type &Key,
              const char *Fmt, Args &&...ArgVals) {
  auto Iter = Map.find(Key);
  if (Iter == Map.end()) {
    fatal(Fmt, std::forward<Args>(ArgVals)...);
  }
  return Iter->second;
}

/// Inserts \p Key into \p Map, reporting a fatal error if it is already
/// present. Used for the write-once records built while resolving a TU.
template <typename MapT, typename... Args>
void insertOrFatal(MapT &Map, const typename MapT::key_type &Key,
                   typename MapT::mapped_type Value, const char *Fmt,
                   Args &&...ArgVals) {
  auto [Iter, Inserted] = Map.insert({Key, std::move(Value)});
  if (!Inserted) {
    fatal(Fmt, std::forward<Args>(ArgVals)...);
  }
}

/// Returns true if \p Linkage makes the entity visible across translation
/// units, and so able to legitimately collide with an occurrence in another TU.
bool isExternal(const EntityLinkage &Linkage) {
  return Linkage.getLinkage() == EntityLinkageType::External;
}

} // namespace

//===----------------------------------------------------------------------===//
// Namespace Resolution
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Linkage Reconciliation
//===----------------------------------------------------------------------===//

EntityBinding
EntityLinker::effectiveBinding(const EntityLinkage &Linkage) const {
  return Rules->effectiveBinding(Linkage);
}

bool EntityLinker::isConflictingDefinition(
    const EntityLinkage &Current, const EntityLinkage &Incoming) const {
  // Only two definitions can conflict; what makes them incompatible is the
  // target's business.
  if (Current.DefinitionKind != EntityDefinitionKind::Definition ||
      Incoming.DefinitionKind != EntityDefinitionKind::Definition) {
    return false;
  }
  return Rules->isConflictingDefinition(Current, Incoming);
}

bool EntityLinker::incomingDataWins(const EntityLinkage &Current,
                                    const EntityLinkage &Incoming) const {
  const bool CurrentDefines =
      Current.DefinitionKind == EntityDefinitionKind::Definition;
  const bool IncomingDefines =
      Incoming.DefinitionKind == EntityDefinitionKind::Definition;

  // A definition beats a declaration.
  if (CurrentDefines != IncomingDefines) {
    return IncomingDefines;
  }

  // Among definitions the stronger binding wins. Two declarations carry no
  // data worth preferring, and equally strong definitions keep the data
  // already linked.
  return IncomingDefines && Rules->strengthRank(effectiveBinding(Incoming)) >
                                Rules->strengthRank(effectiveBinding(Current));
}

EntityLinkage EntityLinker::mergeLinkage(const EntityLinkage &Current,
                                         const EntityLinkage &Incoming) const {
  // Entities collide only when their names, namespaces included, are identical,
  // and namespace resolution qualifies `None` and `Internal` entities with
  // their TU namespace. Occurrences of different linkage therefore never
  // reconcile, and the linkage type carries through unchanged rather than
  // joining.
  assert(Current.Linkage == Incoming.Linkage &&
         "EntityLinker::mergeLinkage: only same-linkage occurrences of an "
         "entity can be reconciled");

  const bool CurrentDefines =
      Current.DefinitionKind == EntityDefinitionKind::Definition;
  const bool IncomingDefines =
      Incoming.DefinitionKind == EntityDefinitionKind::Definition;

  // The binding is the stronger of the two, and the coalescing guarantee holds
  // only if every copy carries it. Both describe definitions, so a declaration
  // contributes neither when the other occurrence defines the entity: on
  // Mach-O, where a weak binding outranks a common one, joining a weak
  // declaration with a common definition would otherwise describe the common
  // definition as weak.
  EntityBinding MergedBinding = Current.Binding;
  EntityCoalescing MergedCoalescing = Current.Coalescing;
  if (CurrentDefines == IncomingDefines) {
    const bool IncomingStronger =
        Rules->strengthRank(effectiveBinding(Incoming)) >
        Rules->strengthRank(effectiveBinding(Current));
    const EntityLinkage &Winner = IncomingStronger ? Incoming : Current;

    MergedCoalescing =
        (CurrentDefines && Current.Coalescing == EntityCoalescing::ODR &&
         Incoming.Coalescing == EntityCoalescing::ODR)
            ? EntityCoalescing::ODR
            : EntityCoalescing::None;

    // A strong binding that only behaved weakly because of its ODR guarantee
    // must degrade along with it. Recording Strong once the guarantee is gone
    // would claim a strong definition the program does not have, and a later
    // genuinely strong definition would then be reported as a duplicate.
    // Doing this also makes the merge commutative when an ODR definition meets
    // an explicitly weak one, since the two tie on effective binding.
    MergedBinding = MergedCoalescing == Winner.Coalescing
                        ? Winner.Binding
                        : effectiveBinding(Winner);
  } else if (IncomingDefines) {
    MergedBinding = Incoming.Binding;
    MergedCoalescing = Incoming.Coalescing;
  }

  return EntityLinkage(Current.Linkage, MergedBinding, MergedCoalescing,
                       Rules->mergeVisibility(Current, Incoming),
                       (CurrentDefines || IncomingDefines)
                           ? EntityDefinitionKind::Definition
                           : EntityDefinitionKind::Declaration);
}

void EntityLinker::reportIfLinkageIsNotExternal(EntityId NewId,
                                                const EntityLinkage &Linkage) {
  if (!isExternal(Linkage)) {
    fatal(ErrorMessages::EntityAlreadyExistsInLinkageTable, NewId, Linkage);
  }
}

bool EntityLinker::reportIfDefinitionsConflict(
    const EntityLinkage &Current, const EntityLinkage &Incoming,
    const EntityName &Name, const NestedBuildNamespace &TUNamespace) const {
  if (!isConflictingDefinition(Current, Incoming)) {
    return false;
  }

  if (!WarnOnMultipleDefinitions) {
    ErrorBuilder::fatal(ErrorMessages::MultipleDefinition, Name, TUNamespace);
  }

  llvm::WithColor::warning()
      << llvm::formatv(ErrorMessages::MultipleDefinition, Name, TUNamespace)
      << "\n";

  return true;
}

void EntityLinker::reportIfNotRepresentable(const EntityLinkage &Linkage,
                                            const EntityName &Name) const {
  if (!Rules->isRepresentable(Linkage)) {
    ErrorBuilder::fatal(ErrorMessages::UnrepresentableLinkage, Name, Linkage,
                        Rules->getName());
  }
}

void EntityLinker::reportIfOrderDependent(const EntityLinkage &Current,
                                          const EntityLinkage &Incoming,
                                          const EntityName &Name) const {
  if (Rules->isOrderDependentMerge(Current, Incoming)) {
    llvm::WithColor::warning()
        << llvm::formatv(ErrorMessages::OrderDependentMerge, Name) << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Resolve
//===----------------------------------------------------------------------===//

std::pair<EntityId, bool>
EntityLinker::resolveEntity(const EntityName &OldName,
                            const EntityLinkage &RawLinkage,
                            const NestedBuildNamespace &TUNamespace) {
  NestedBuildNamespace NewNamespace =
      resolveNamespace(Output.LUNamespace, TUNamespace, OldName.Namespace,
                       RawLinkage.getLinkage());

  EntityName NewName(OldName.USR, OldName.Suffix, NewNamespace);

  // The summary records what the source declared; coerce it to what this
  // target can express before any rule looks at it, and reject values the
  // target's toolchain could never have produced.
  reportIfNotRepresentable(RawLinkage, NewName);
  const EntityLinkage Linkage = Rules->normalize(RawLinkage);

  // NewId construction will always return a fresh id for `None` and `Internal`
  // linkage entities since their namespaces will be different even if their
  // names clash. For `External` linkage entities with identical names this
  // function will return the id assigned at the first insertion.
  EntityId NewId = Output.IdTable.getId(NewName);

  auto [It, Inserted] = Output.LinkageTable.try_emplace(NewId, Linkage);
  if (Inserted) {
    // Keep the name so finalization can name the entity in diagnostics.
    EntityNames.insert({NewId, NewName});
    // First occurrence of this entity: its summary data is the copy to keep.
    return {NewId, true};
  }

  reportIfLinkageIsNotExternal(NewId, Linkage);

  // `External` entities collide legitimately. A conflicting duplicate
  // definition is ignored outright, leaving the definition already linked in
  // place; anything else is reconciled, and the data selection tells merge()
  // whose encoding to keep.
  if (reportIfDefinitionsConflict(It->second, Linkage, NewName, TUNamespace)) {
    return {NewId, false};
  }

  reportIfOrderDependent(It->second, Linkage, NewName);

  const bool DataFromIncoming = incomingDataWins(It->second, Linkage);
  It->second = mergeLinkage(It->second, Linkage);

  return {NewId, DataFromIncoming};
}

std::pair<EntityResolutionMap, EntityLinker::DataSelectionMap>
EntityLinker::resolve(const TUSummaryEncoding &Summary) {
  EntityResolutionMap Resolution;
  DataSelectionMap DataSelection;

  Summary.IdTable.forEach([&](const EntityName &OldName, const EntityId OldId) {
    const EntityLinkage &Linkage =
        lookupOrFatal(Summary.LinkageTable, OldId,
                      ErrorMessages::MissingLinkageInformation, OldId);

    auto [NewId, DataFromIncoming] = resolveEntity(
        OldName, Linkage, NestedBuildNamespace(Summary.TUNamespace));

    insertOrFatal(Resolution, OldId, NewId,
                  ErrorMessages::DuplicateEntityIdInTUSummary, OldId);
    insertOrFatal(DataSelection, NewId, DataFromIncoming,
                  ErrorMessages::MultipleEntitiesResolveToSameId, NewId);
  });

  return {std::move(Resolution), std::move(DataSelection)};
}

//===----------------------------------------------------------------------===//
// Merge
//===----------------------------------------------------------------------===//

EntitySummaryEncoding *EntityLinker::mergeEntityData(
    EntityDataMap &OutputData, const SummaryName &SN, EntityId NewId,
    std::unique_ptr<EntitySummaryEncoding> &Encoding,
    const EntityLinkage &Linkage, bool DataFromIncoming) {
  auto [It, Inserted] = OutputData.try_emplace(NewId, std::move(Encoding));

  // No earlier TU contributed data for this entity, so this encoding is it.
  if (Inserted) {
    return It->second.get();
  }

  // Insertion should never fail for `None` and `Internal` linkage entities
  // because these entities will have different namespaces across TUs even if
  // their names match.
  if (!isExternal(Linkage)) {
    fatal(ErrorMessages::FailedToInsertEntityIntoOutputSummary, NewId, Linkage,
          SN);
  }

  // Two definitions required to be identical should carry identical summaries.
  // Keep the loser so reportODRMismatches() can compare them once both are in
  // the LU EntityId space; a displaced encoding from this TU still needs
  // patching, so it is returned as a patch target either way.
  const bool CompareDisplaced = ODRMismatch != ODRMismatchPolicy::Ignore &&
                                Linkage.Coalescing == EntityCoalescing::ODR;

  // `External` collision: the reconciliation in resolve() decided which
  // occurrence's data is the copy to keep. If this TU won, its encoding
  // replaces the incumbent; otherwise the incumbent is kept and this encoding
  // is dropped.
  if (!DataFromIncoming) {
    if (!CompareDisplaced) {
      return nullptr;
    }
    Displaced.push_back({SN, NewId, std::move(Encoding)});
    return Displaced.back().Data.get();
  }

  if (CompareDisplaced) {
    // The incumbent is already patched, so it needs no further work.
    Displaced.push_back({SN, NewId, std::move(It->second)});
  }
  It->second = std::move(Encoding);
  return It->second.get();
}

std::vector<EntitySummaryEncoding *>
EntityLinker::mergeSummaryData(const SummaryName &SN, EntityDataMap &DataMap,
                               const TUSummaryEncoding &Summary,
                               const EntityResolutionMap &Resolution,
                               const DataSelectionMap &DataSelection) {
  std::vector<EntitySummaryEncoding *> PatchTargets;
  auto &OutputSummaryData = Output.Data[SN];

  for (auto &[OldId, ES] : DataMap) {
    const EntityId NewId =
        lookupOrFatal(Resolution, OldId,
                      ErrorMessages::EntityNotFoundInResolutionTable, OldId);

    // Safe to retrieve linkage and data selection using .at since the resolve
    // step records both for every OldId in the TU summary.
    assert(Summary.LinkageTable.count(OldId) &&
           "EntityLinker::mergeSummaryData: resolve() records a linkage for "
           "every entity");
    assert(DataSelection.count(NewId) &&
           "EntityLinker::mergeSummaryData: resolve() records a data selection "
           "for every resolved entity");

    if (auto *PatchTarget = mergeEntityData(OutputSummaryData, SN, NewId, ES,
                                            Summary.LinkageTable.at(OldId),
                                            DataSelection.at(NewId))) {
      PatchTargets.push_back(PatchTarget);
    }
  }

  return PatchTargets;
}

std::vector<EntitySummaryEncoding *>
EntityLinker::merge(TUSummaryEncoding &Summary,
                    const EntityResolutionMap &Resolution,
                    const DataSelectionMap &DataSelection) {
  std::vector<EntitySummaryEncoding *> PatchTargets;

  for (auto &[SN, DataMap] : Summary.Data) {
    std::vector<EntitySummaryEncoding *> SummaryTargets =
        mergeSummaryData(SN, DataMap, Summary, Resolution, DataSelection);
    PatchTargets.insert(PatchTargets.end(), SummaryTargets.begin(),
                        SummaryTargets.end());
  }

  return PatchTargets;
}

//===----------------------------------------------------------------------===//
// Patch
//===----------------------------------------------------------------------===//

llvm::Error
EntityLinker::patch(const std::vector<EntitySummaryEncoding *> &PatchTargets,
                    const EntityResolutionMap &Resolution) {
  for (auto *PatchTarget : PatchTargets) {
    assert(PatchTarget && "EntityLinker::patch: Patch target cannot be null");

    if (auto Err = PatchTarget->patch(Resolution)) {
      return Err;
    }
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Link
//===----------------------------------------------------------------------===//

llvm::Error
EntityLinker::checkTUNotAlreadyLinked(const BuildNamespace &TUNamespace) {
  auto [_, Inserted] = ProcessedTUNamespaces.insert(TUNamespace);
  if (!Inserted) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::DuplicateTUNamespace,
                                TUNamespace)
        .build();
  }
  return llvm::Error::success();
}

llvm::Error
EntityLinker::checkTUTargetMatches(const llvm::Triple &TUTriple,
                                   const BuildNamespace &TUNamespace) const {
  // Compare only what selects resolution rules and affects the outcome. The OS
  // version does not, so macosx14.0 and macosx15.0 are interchangeable here,
  // matching the key MultiArch* uses to order its members.
  const llvm::Triple &LUTriple = Output.TargetTriple;
  const bool Matches = TUTriple.getArch() == LUTriple.getArch() &&
                       TUTriple.getSubArch() == LUTriple.getSubArch() &&
                       TUTriple.getVendor() == LUTriple.getVendor() &&
                       TUTriple.getOS() == LUTriple.getOS() &&
                       TUTriple.getEnvironment() == LUTriple.getEnvironment() &&
                       TUTriple.getObjectFormat() == LUTriple.getObjectFormat();
  if (Matches) {
    return llvm::Error::success();
  }

  return ErrorBuilder::create(std::errc::invalid_argument,
                              ErrorMessages::TargetTripleMismatch, TUNamespace,
                              TUTriple.str(),
                              LinkageRules::forTarget(TUTriple).getName(),
                              LUTriple.str(), Rules->getName())
      .build();
}

llvm::Error EntityLinker::link(std::unique_ptr<TUSummaryEncoding> Summary) {
  if (auto Err = checkTUNotAlreadyLinked(Summary->TUNamespace)) {
    return Err;
  }

  if (auto Err = checkTUTargetMatches(Summary->getTargetTriple(),
                                      Summary->TUNamespace)) {
    return Err;
  }

  TUSummaryEncoding &SummaryRef = *Summary;
  Displaced.clear();

  auto [Resolution, DataSelection] = resolve(SummaryRef);
  auto PatchTargets = merge(SummaryRef, Resolution, DataSelection);
  if (auto Err = patch(PatchTargets, Resolution)) {
    return Err;
  }

  reportODRMismatches(NestedBuildNamespace(SummaryRef.TUNamespace));
  return llvm::Error::success();
}

void EntityLinker::reportODRMismatches(
    const NestedBuildNamespace &TUNamespace) {
  std::vector<std::string> Messages;
  for (const DisplacedData &D : Displaced) {
    const auto &Kept = Output.Data.at(D.Name).at(D.Id);
    if (Kept->equals(*D.Data)) {
      continue;
    }
    auto NameIt = EntityNames.find(D.Id);
    assert(NameIt != EntityNames.end() &&
           "EntityLinker: every resolved entity records its name");
    Messages.push_back(llvm::formatv(ErrorMessages::ODRMismatch, D.Name,
                                     NameIt->second, TUNamespace)
                           .str());
  }
  Displaced.clear();

  if (Messages.empty()) {
    return;
  }

  if (ODRMismatch == ODRMismatchPolicy::Warn) {
    for (const std::string &Message : Messages) {
      llvm::WithColor::warning() << Message << "\n";
    }
    return;
  }

  ErrorBuilder::fatal("{0}", llvm::join(Messages, "\n"));
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

void EntityLinker::reportUnresolvedEntities() const {
  if (UnresolvedSymbols == UnresolvedPolicy::Ignore) {
    return;
  }

  std::vector<std::string> Messages;
  for (const auto &[Id, Linkage] : Output.LinkageTable) {
    // Only an external entity can be satisfied by another translation unit.
    if (Linkage.Linkage != EntityLinkageType::External ||
        Linkage.DefinitionKind == EntityDefinitionKind::Definition) {
      continue;
    }
    // An undefined weak reference resolves to nothing by design.
    if (Rules->effectiveBinding(Linkage) == EntityBinding::Weak) {
      continue;
    }
    auto NameIt = EntityNames.find(Id);
    assert(NameIt != EntityNames.end() &&
           "EntityLinker: every resolved entity records its name");
    Messages.push_back(
        llvm::formatv(ErrorMessages::UnresolvedEntity, NameIt->second).str());
  }

  if (Messages.empty()) {
    return;
  }

  if (UnresolvedSymbols == UnresolvedPolicy::Warn) {
    for (const std::string &Message : Messages) {
      llvm::WithColor::warning() << Message << "\n";
    }
    return;
  }

  ErrorBuilder::fatal("{0}", llvm::join(Messages, "\n"));
}

void EntityLinker::demoteHiddenEntities() {
  for (auto &[Id, Linkage] : Output.LinkageTable) {
    if (Linkage.Linkage != EntityLinkageType::External) {
      continue;
    }
    // "Hidden" is whatever the target treats as most restrictive; on COFF,
    // where visibility is meaningless, normalize() has already flattened
    // everything to Default so nothing is demoted.
    if (Linkage.Visibility != EntityVisibility::Hidden) {
      continue;
    }
    Linkage = EntityLinkage(EntityLinkageType::Internal, Linkage.Binding,
                            Linkage.Coalescing, Linkage.Visibility,
                            Linkage.DefinitionKind);
  }
}

void EntityLinker::finalize() {
  reportUnresolvedEntities();
  demoteHiddenEntities();
}
