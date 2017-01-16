//===-- ApplyReplacements.h - Deduplicate and apply replacements -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the interface for deduplicating, detecting
/// conflicts in, and applying collections of Replacements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APPLYREPLACEMENTS_H
#define LLVM_CLANG_APPLYREPLACEMENTS_H

#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <system_error>
#include <vector>

namespace clang {

class DiagnosticsEngine;
class Rewriter;

namespace format {
struct FormatStyle;
} // end namespace format

namespace replace {

/// \brief Collection of source ranges.
typedef std::vector<clang::tooling::Range> RangeVector;

/// \brief Collection of TranslationUnitReplacements.
typedef std::vector<clang::tooling::TranslationUnitReplacements> TUReplacements;

/// \brief Collection of TranslationUnitReplacement files.
typedef std::vector<std::string> TUReplacementFiles;

/// \brief Collection of TranslationUniDiagnostics.
typedef std::vector<clang::tooling::TranslationUnitDiagnostics> TUDiagnostics;

/// \brief Map mapping file name to Replacements targeting that file.
typedef llvm::DenseMap<const clang::FileEntry *,
                       std::vector<clang::tooling::Replacement>>
    FileToReplacementsMap;

/// \brief Recursively descends through a directory structure rooted at \p
/// Directory and attempts to deserialize *.yaml files as
/// TranslationUnitReplacements. All docs that successfully deserialize are
/// added to \p TUs.
///
/// Directories starting with '.' are ignored during traversal.
///
/// \param[in] Directory Directory to begin search for serialized
/// TranslationUnitReplacements.
/// \param[out] TUs Collection of all found and deserialized
/// TranslationUnitReplacements or TranslationUnitDiagnostics.
/// \param[out] TUFiles Collection of all TranslationUnitReplacement files
/// found in \c Directory.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns An error_code indicating success or failure in navigating the
/// directory structure.
std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUReplacements &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics);

std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUDiagnostics &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics);

/// \brief Deduplicate, check for conflicts, and apply all Replacements stored
/// in \c TUs. If conflicts occur, no Replacements are applied.
///
/// \post For all (key,value) in GroupedReplacements, value[i].getOffset() <=
/// value[i+1].getOffset().
///
/// \param[in] TUs Collection of TranslationUnitReplacements or
/// TranslationUnitDiagnostics to merge,
/// deduplicate, and test for conflicts.
/// \param[out] GroupedReplacements Container grouping all Replacements by the
/// file they target.
/// \param[in] SM SourceManager required for conflict reporting.
///
/// \returns \parblock
///          \li true If all changes were applied successfully.
///          \li false If there were conflicts.
bool mergeAndDeduplicate(const TUReplacements &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::SourceManager &SM);

bool mergeAndDeduplicate(const TUDiagnostics &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::SourceManager &SM);

// FIXME: Remove this function after changing clang-apply-replacements to use
// Replacements class.
bool applyAllReplacements(const std::vector<tooling::Replacement> &Replaces,
                          Rewriter &Rewrite);

/// \brief Apply all replacements in \c GroupedReplacements.
///
/// \param[in] GroupedReplacements Deduplicated and conflict free Replacements
/// to apply.
/// \param[out] Rewrites The results of applying replacements will be applied
/// to this Rewriter.
///
/// \returns \parblock
///          \li true If all changes were applied successfully.
///          \li false If a replacement failed to apply.
bool applyReplacements(const FileToReplacementsMap &GroupedReplacements,
                       clang::Rewriter &Rewrites);

/// \brief Given a collection of Replacements for a single file, produces a list
/// of source ranges that enclose those Replacements.
///
/// \pre Replacements[i].getOffset() <= Replacements[i+1].getOffset().
///
/// \param[in] Replacements Replacements from a single file.
///
/// \returns Collection of source ranges that enclose all given Replacements.
/// One range is created for each replacement.
RangeVector calculateChangedRanges(
    const std::vector<clang::tooling::Replacement> &Replacements);

/// \brief Write the contents of \c FileContents to disk. Keys of the map are
/// filenames and values are the new contents for those files.
///
/// \param[in] Rewrites Rewriter containing written files to write to disk.
bool writeFiles(const clang::Rewriter &Rewrites);

/// \brief Delete the replacement files.
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
