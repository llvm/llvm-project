//===- LinkageRules.h - Per-platform symbol resolution ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines LinkageRules, which supplies the symbol resolution rules
//  the entity linker emulates for a given object format. Every rule here is
//  grounded in an observed linker behaviour; see
//  docs/ssaf-linker-{elf,macho,coff}-behavior.md.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LINKAGERULES_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LINKAGERULES_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace clang::ssaf {

/// The platform-specific symbol resolution rules the linker emulates.
///
/// The EntityLinkage enums carry no ordering of their own, because precedence
/// is platform-specific: Mach-O ranks a weak definition above a common one
/// while ELF and COFF do the reverse, and Mach-O merges visibility to the
/// least restrictive while ELF merges to the most. The ranks below answer
/// "which value wins the join", not "which value is semantically greater", and
/// are deliberately inverted for some fields on some targets.
class LinkageRules {
public:
  virtual ~LinkageRules() = default;

  /// Returns the rules for \p TargetTriple, dispatching on its object format.
  ///
  /// The returned reference has static storage duration. An unsupported object
  /// format is a fatal error naming the triple.
  static const LinkageRules &forTarget(const llvm::Triple &TargetTriple);

  /// The object format these rules emulate, for diagnostics.
  virtual llvm::StringRef getName() const = 0;

  /// Lowers a source-level linkage to the binding this platform actually
  /// emits.
  ///
  /// A summary records what the source declared, so an inline function is
  /// Strong + ODR regardless of target. The object formats disagree about how
  /// to express that, and the disagreement changes resolution, so every
  /// comparison below is made on the lowered value rather than the raw one.
  virtual EntityBinding
  effectiveBinding(const EntityLinkage &Linkage) const = 0;

  /// Precedence among bindings: the greater rank prevails when two definitions
  /// of the same entity collide.
  ///
  /// \param B A binding already lowered by effectiveBinding().
  virtual unsigned strengthRank(EntityBinding B) const = 0;

  /// Precedence among visibilities: the greater rank prevails when two
  /// occurrences of the same entity disagree.
  virtual unsigned visibilityRank(EntityVisibility V) const = 0;

  /// Returns true if two definitions of the same external entity cannot
  /// coexist, which is the multiple-definition error a real linker reports.
  ///
  /// This is not derivable from the ranks: COFF licenses duplicate definitions
  /// by COMDAT rather than by weakness, and rejects two weak definitions that
  /// ELF and Mach-O both accept.
  ///
  /// \pre Both occurrences are definitions.
  virtual bool isConflictingDefinition(const EntityLinkage &Current,
                                       const EntityLinkage &Incoming) const = 0;

  /// Merges the visibility of two occurrences of the same entity.
  ///
  /// Defaults to the higher visibilityRank(). Mach-O overrides this because
  /// its rule for common symbols depends on the bindings, not just the
  /// visibilities.
  virtual EntityVisibility mergeVisibility(const EntityLinkage &Current,
                                           const EntityLinkage &Incoming) const;

  /// Returns true if merging these two occurrences gives a result that depends
  /// on which was linked first, so the caller can warn that the program is
  /// ambiguous.
  virtual bool isOrderDependentMerge(const EntityLinkage &Current,
                                     const EntityLinkage &Incoming) const;

  /// Normalizes \p Linkage to what this platform can represent.
  ///
  /// Values the platform's toolchain silently drops are coerced to their
  /// platform equivalent. Values the toolchain could never have produced are
  /// left for isRepresentable() to reject, since they indicate a corrupted or
  /// hand-edited summary rather than ordinary portable source.
  virtual EntityLinkage normalize(const EntityLinkage &Linkage) const;

  /// Returns false if \p Linkage holds a value this platform's toolchain could
  /// not have produced, which the caller reports as a fatal error.
  ///
  /// Distinct from normalize(): a value that is merely dropped at emission is
  /// coerced, whereas a value clang refuses to emit at all is rejected.
  virtual bool isRepresentable(const EntityLinkage &Linkage) const;

protected:
  // Friendship with EntityLinkage does not extend to subclasses, so the base
  // reads its private fields on their behalf.
  static EntityLinkageType linkageOf(const EntityLinkage &L);
  static EntityBinding bindingOf(const EntityLinkage &L);
  static EntityCoalescing coalescingOf(const EntityLinkage &L);
  static EntityVisibility visibilityOf(const EntityLinkage &L);

  /// True when \p Linkage defines the entity rather than declaring it.
  static bool defines(const EntityLinkage &Linkage);

  /// Builds a copy of \p Linkage with its visibility replaced.
  static EntityLinkage withVisibility(const EntityLinkage &Linkage,
                                      EntityVisibility V);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_ENTITYLINKER_LINKAGERULES_H
