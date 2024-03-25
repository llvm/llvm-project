//===-- ApplyReplacements.h - Deduplicate and apply replacements -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the interface for deduplicating, detecting
/// conflicts in, and applying collections of Replacements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APPLYREPLACEMENTS_H
#define LLVM_CLANG_APPLYREPLACEMENTS_H

#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <system_error>
#include <vector>

namespace clang {

class DiagnosticsEngine;

namespace replace {

/// Collection of TranslationUnitReplacements.
typedef std::vector<clang::tooling::TranslationUnitReplacements> TUReplacements;

/// Collection of TranslationUnitReplacement files.
typedef std::vector<std::string> TUReplacementFiles;

/// Collection of TranslationUniDiagnostics.
typedef std::vector<clang::tooling::TranslationUnitDiagnostics> TUDiagnostics;

/// Map mapping file name to a set of AtomicChange targeting that file.
using FileToChangesMap =
    llvm::DenseMap<clang::FileEntryRef, std::vector<tooling::AtomicChange>>;

/// Recursively descends through a directory structure rooted at \p
/// Directory and attempts to deserialize *.yaml files as
/// TranslationUnitReplacements. All docs that successfully deserialize are
/// added to \p TUs.
///
/// Directories starting with '.' are ignored during traversal.
///
/// \param[in] Directory Directory to begin search for serialized
/// TranslationUnitReplacements or TranslationUnitDiagnostics.
/// \param[out] TUs Collection of all found and deserialized
/// TranslationUnitReplacements or TranslationUnitDiagnostics.
/// \param[out] TUFiles Collection of all TranslationUnitReplacement or
/// TranslationUnitDiagnostics files found in \c Directory.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns An error_code indicating success or failure in navigating the
/// directory structure.
template <typename TranslationUnits>
std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TranslationUnits &TUs,
    TUReplacementFiles &TUFiles,
    clang::DiagnosticsEngine &Diagnostics) = delete;

template <>
std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUReplacements &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics);

template <>
std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUDiagnostics &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics);

/// Deduplicate, check for conflicts, and extract all Replacements stored
/// in \c TUs. Conflicting replacements are skipped.
///
/// \post For all (key,value) in FileChanges, value[i].getOffset() <=
/// value[i+1].getOffset().
///
/// \param[in] TUs Collection of TranslationUnitReplacements or
/// TranslationUnitDiagnostics to merge, deduplicate, and test for conflicts.
/// \param[out] FileChanges Container grouping all changes by the
/// file they target. Only non conflicting replacements are kept into
/// FileChanges.
/// \param[in] SM SourceManager required for conflict reporting.
///
/// \returns \parblock
///          \li true If all changes were converted successfully.
///          \li false If there were conflicts.
bool mergeAndDeduplicate(const TUReplacements &TUs, const TUDiagnostics &TUDs,
                         FileToChangesMap &FileChanges,
                         clang::SourceManager &SM,
                         bool IgnoreInsertConflict = false);

/// Apply \c AtomicChange on File and rewrite it.
///
/// \param[in] File Path of the file where to apply AtomicChange.
/// \param[in] Changes to apply.
/// \param[in] Spec For code cleanup and formatting.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns The changed code if all changes are applied successfully;
/// otherwise, an llvm::Error carrying llvm::StringError or an error_code.
llvm::Expected<std::string>
applyChanges(StringRef File, const std::vector<tooling::AtomicChange> &Changes,
             const tooling::ApplyChangesSpec &Spec,
             DiagnosticsEngine &Diagnostics);

/// Delete the replacement files.
///
/// \param[in] Files Replacement files to delete.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns \parblock
///          \li true If all files have been deleted successfully.
///          \li false If at least one or more failures occur when deleting
/// files.
bool deleteReplacementFiles(const TUReplacementFiles &Files,
                            clang::DiagnosticsEngine &Diagnostics);

} // end namespace replace
} // end namespace clang

#endif // LLVM_CLANG_APPLYREPLACEMENTS_H
