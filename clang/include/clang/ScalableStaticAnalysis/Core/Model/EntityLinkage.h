//===- EntityLinkage.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_ENTITYLINKAGE_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_ENTITYLINKAGE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace clang::ssaf {

/// Symbol scope.
///
/// Enumerator order is insignificant: the linker never compares these values,
/// it switches on them by name.
enum class EntityLinkageType : uint8_t {
  None,     ///< local variables
  Internal, ///< static functions/variables, anonymous namespace
  External  ///< globally visible across translation units (including parameters
            ///< of functions with external linkage)
};

/// Symbol strength: which definition prevails when two of them collide.
///
/// Enumerator order is insignificant. Precedence is platform-specific — on
/// Mach-O a weak definition displaces a common, while on ELF and COFF the
/// common wins — so comparisons must go through LinkageRules::strengthRank()
/// rather than comparing these values directly.
enum class EntityBinding : uint8_t {
  Undefined, ///< a reference with no definition in this unit
  Weak,      ///< __attribute__((weak)); overridden by a strong definition
  Common,    ///< C tentative definition; merged, size = max
  Strong     ///< ordinary definition
};

/// How far a symbol escapes its link unit.
///
/// Enumerator order is insignificant. ELF merges to the most restrictive
/// visibility, Mach-O to the least, and COFF has no visibility at all; see
/// LinkageRules::mergeVisibility().
enum class EntityVisibility : uint8_t {
  Default,  ///< participates in cross-LU resolution; exported/interposable
  Hidden,   ///< visible within the LU, not exported past its boundary
  Protected ///< exported but bound to the defining LU (not interposable)
};

/// Whether every definition of the entity is required to be identical, as a
/// COMDAT group (ELF and COFF) or a .weak_definition (Mach-O) guarantees.
///
/// This is independent of EntityBinding, because the object formats encode the
/// two properties separately: an inline function is emitted weak on ELF and
/// Mach-O but strong on COFF, and in every case the ODR guarantee is carried by
/// a mechanism other than the binding. See LinkageRules::effectiveBinding().
///
/// Modelled as an enum rather than a bool so that COFF's other
/// IMAGE_COMDAT_SELECT_* kinds can be added without a schema change.
enum class EntityCoalescing : uint8_t {
  None, ///< definitions are unrelated; at most one may exist
  ODR   ///< all definitions are required to be identical and are coalesced
};

/// Whether this occurrence defines the entity or merely declares/references it.
///
/// Enumerator order is insignificant; see
/// LinkageRules::definitionKindRank().
enum class EntityDefinitionKind : uint8_t {
  Declaration, ///< declares or references the entity
  Definition   ///< defines the entity
};

/// Represents the linker-relevant properties of an entity in the program model.
///
/// EntityLinkage captures the scope, strength, coalescing, visibility, and
/// definition state of an entity, which together determine how it is resolved
/// and exported across translation units and link units.
///
/// The values are recorded as the source declares them; translating them into
/// what a particular object format would emit is the linker's job, via
/// LinkageRules.
class EntityLinkage {
  friend class EntityLinker;
  friend class LinkageRules;
  friend class SerializationFormat;
  friend class TestFixture;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const EntityLinkage &Linkage);

public:
  constexpr EntityLinkage(EntityLinkageType Linkage, EntityBinding Binding,
                          EntityCoalescing Coalescing,
                          EntityVisibility Visibility,
                          EntityDefinitionKind DefinitionKind)
      : Linkage(Linkage), Binding(Binding), Coalescing(Coalescing),
        Visibility(Visibility), DefinitionKind(DefinitionKind) {}

  EntityLinkageType getLinkage() const { return Linkage; }

  bool operator==(const EntityLinkage &Other) const;
  bool operator!=(const EntityLinkage &Other) const;

private:
  EntityLinkageType Linkage;
  EntityBinding Binding;
  EntityCoalescing Coalescing;
  EntityVisibility Visibility;
  EntityDefinitionKind DefinitionKind;
};

/// Writes a string representation of the linkage type to the stream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, EntityLinkageType Linkage);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, EntityBinding Binding);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              EntityCoalescing Coalescing);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              EntityVisibility Visibility);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              EntityDefinitionKind DefinitionKind);

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const EntityLinkage &Linkage);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_ENTITYLINKAGE_H
