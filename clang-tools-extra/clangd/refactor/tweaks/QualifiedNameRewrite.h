//===--- QualifiedNameRewrite.h ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for tweaks that rewrite namespace-qualified references in the
// main file, such as AddUsingReplaceAll and ReplaceQualifiedWithAlias. These
// tweaks differ only in the replacement they emit; the logic for locating and
// filtering the qualified references to rewrite is identical and lives here.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_TWEAKS_QUALIFIEDNAMEREWRITE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_TWEAKS_QUALIFIEDNAMEREWRITE_H

#include "FindTarget.h"
#include "refactor/Tweak.h"

#include "clang/AST/Decl.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {
namespace clangd {

/// Returns the canonical NamedDecl for \p D, or nullptr.
const NamedDecl *canonicalDecl(const NamedDecl *D);

/// True if \p Qualifier is a namespace qualifier (e.g. 'ns::').
bool isNamespaceQualifier(NestedNameSpecifierLoc Qualifier);

/// Returns the NamespaceDecl behind a namespace qualifier, or nullptr if
/// \p Qualifier is not a (valid) namespace qualifier.
const NamespaceDecl *getNamespaceFromQualifier(NestedNameSpecifier Qualifier);

/// Normalizes \p Loc to a location written in the main file.
///
/// References inside macro arguments are mapped to their spelling location via
/// getFileLoc(); macro-body expansions and locations outside the main file are
/// rejected. Returns true and updates \p Loc on success, false otherwise.
bool normalizeToMainFile(SourceLocation &Loc, const SourceManager &SM);

/// True if any of \p Ref's targets canonically matches an entry in \p Targets.
bool referenceMatchesTargets(const ReferenceLoc &Ref,
                             const llvm::DenseSet<const NamedDecl *> &Targets);

/// Returns the end of a qualified reference starting at \p NameLoc, extending
/// past any template argument list so rewrites replace the full spelled name.
SourceLocation getReferenceEnd(SourceLocation NameLoc, const SourceManager &SM,
                               const LangOptions &LangOpts);

/// A namespace-qualified reference that passed all rewrite filters, with its
/// locations already normalized to the main file.
struct QualifiedReference {
  /// Start of the qualifier, e.g. 'ns::' in 'ns::foo'.
  SourceLocation QualifierLoc;
  /// Start of the last name part, e.g. 'foo' in 'ns::foo<int>'.
  SourceLocation NameLoc;
};

/// Walks every local top-level declaration and invokes \p Callback for each
/// explicit, namespace-qualified reference (not a declaration) that resolves to
/// one of \p Targets. Locations passed to \p Callback are normalized to the
/// main file. The callback decides which replacement, if any, to emit.
void forEachQualifiedReferenceToRewrite(
    const Tweak::Selection &Inputs,
    const llvm::DenseSet<const NamedDecl *> &Targets,
    llvm::function_ref<void(const QualifiedReference &)> Callback);

enum class QualifiedRewriteClient : unsigned {
  AddUsingReplaceAll = 1 << 0,
  ReplaceQualifiedWithAlias = 1 << 1,
};

/// Result of analyzing a Tweak::Selection for namespace-qualified reference
/// rewrite tweaks (AddUsingReplaceAll and ReplaceQualifiedWithAlias).
struct PreparedQualifiedSelection {
  /// Enclosing TypeAliasDecl or UsingDecl when the selection is inside an alias
  /// declaration.
  const NamedDecl *SelectedAliasDecl = nullptr;

  /// True if the selection is on a valid namespace-qualified reference in the
  /// main file outside macro bodies.
  bool HasQualifiedRef = false;
  NestedNameSpecifierLoc QualifierToRemove;
  std::string QualifierToSpell;
  llvm::StringRef SpelledQualifier;
  llvm::StringRef SpelledName;
  SourceLocation MustInsertAfterLoc;
  llvm::DenseSet<const NamedDecl *> Targets;
};

/// Analyzes \p Inputs once per Tweak::Selection and caches the result so that
/// AddUsingReplaceAll::prepare() and ReplaceQualifiedWithAlias::prepare() share
/// a single execution for the same selection.
const PreparedQualifiedSelection &
prepareQualifiedSelection(const Tweak::Selection &Inputs,
                          QualifiedRewriteClient Client);

} // namespace clangd
} // namespace clang

#endif
