//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various code that examines C++ source code without using heavy AST machinery
// (and often not even the lexer). To be used sparingly!
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SOURCECODE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SOURCECODE_H
#include "Context.h"
#include "Protocol.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"

namespace clang {
class SourceManager;

namespace clangd {

// We tend to generate digests for source codes in a lot of different places.
// This represents the type for those digests to prevent us hard coding details
// of hashing function at every place that needs to store this information.
using FileDigest = decltype(llvm::SHA1::hash({}));
FileDigest digest(StringRef Content);
Optional<FileDigest> digestFile(const SourceManager &SM, FileID FID);

// This context variable controls the behavior of functions in this file
// that convert between LSP offsets and native clang byte offsets.
// If not set, defaults to UTF-16 for backwards-compatibility.
extern Key<OffsetEncoding> kCurrentOffsetEncoding;

// Counts the number of UTF-16 code units needed to represent a string (LSP
// specifies string lengths in UTF-16 code units).
// Use of UTF-16 may be overridden by kCurrentOffsetEncoding.
size_t lspLength(StringRef Code);

/// Turn a [line, column] pair into an offset in Code.
///
/// If P.character exceeds the line length, returns the offset at end-of-line.
/// (If !AllowColumnsBeyondLineLength, then returns an error instead).
/// If the line number is out of range, returns an error.
///
/// The returned value is in the range [0, Code.size()].
llvm::Expected<size_t>
positionToOffset(llvm::StringRef Code, Position P,
                 bool AllowColumnsBeyondLineLength = true);

/// Turn an offset in Code into a [line, column] pair.
/// The offset must be in range [0, Code.size()].
Position offsetToPosition(llvm::StringRef Code, size_t Offset);

/// Turn a SourceLocation into a [line, column] pair.
/// FIXME: This should return an error if the location is invalid.
Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc);

/// Return the file location, corresponding to \p P. Note that one should take
/// care to avoid comparing the result with expansion locations.
llvm::Expected<SourceLocation> sourceLocationInMainFile(const SourceManager &SM,
                                                        Position P);

/// Turns a token range into a half-open range and checks its correctness.
/// The resulting range will have only valid source location on both sides, both
/// of which are file locations.
///
/// File locations always point to a particular offset in a file, i.e. they
/// never refer to a location inside a macro expansion. Turning locations from
/// macro expansions into file locations is ambiguous - one can use
/// SourceManager::{getExpansion|getFile|getSpelling}Loc. This function
/// calls SourceManager::getFileLoc on both ends of \p R to do the conversion.
///
/// User input (e.g. cursor position) is expressed as a file location, so this
/// function can be viewed as a way to normalize the ranges used in the clang
/// AST so that they are comparable with ranges coming from the user input.
llvm::Optional<SourceRange> toHalfOpenFileRange(const SourceManager &Mgr,
                                                const LangOptions &LangOpts,
                                                SourceRange R);

/// Returns true iff all of the following conditions hold:
///   - start and end locations are valid,
///   - start and end locations are file locations from the same file
///     (i.e. expansion locations are not taken into account).
///   - start offset <= end offset.
/// FIXME: introduce a type for source range with this invariant.
bool isValidFileRange(const SourceManager &Mgr, SourceRange R);

/// Returns true iff \p L is contained in \p R.
/// EXPECTS: isValidFileRange(R) == true, L is a file location.
bool halfOpenRangeContains(const SourceManager &Mgr, SourceRange R,
                           SourceLocation L);

/// Returns true iff \p L is contained in \p R or \p L is equal to the end point
/// of \p R.
/// EXPECTS: isValidFileRange(R) == true, L is a file location.
bool halfOpenRangeTouches(const SourceManager &Mgr, SourceRange R,
                          SourceLocation L);

/// Returns the source code covered by the source range.
/// EXPECTS: isValidFileRange(R) == true.
llvm::StringRef toSourceCode(const SourceManager &SM, SourceRange R);

// Converts a half-open clang source range to an LSP range.
// Note that clang also uses closed source ranges, which this can't handle!
Range halfOpenToRange(const SourceManager &SM, CharSourceRange R);

// Converts an offset to a clang line/column (1-based, columns are bytes).
// The offset must be in range [0, Code.size()].
// Prefer to use SourceManager if one is available.
std::pair<size_t, size_t> offsetToClangLineColumn(llvm::StringRef Code,
                                                  size_t Offset);

/// From "a::b::c", return {"a::b::", "c"}. Scope is empty if there's no
/// qualifier.
std::pair<llvm::StringRef, llvm::StringRef>
splitQualifiedName(llvm::StringRef QName);

TextEdit replacementToEdit(StringRef Code, const tooling::Replacement &R);

std::vector<TextEdit> replacementsToEdits(StringRef Code,
                                          const tooling::Replacements &Repls);

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L);

/// Get the canonical path of \p F.  This means:
///
///   - Absolute path
///   - Symlinks resolved
///   - No "." or ".." component
///   - No duplicate or trailing directory separator
///
/// This function should be used when paths needs to be used outside the
/// component that generate it, so that paths are normalized as much as
/// possible.
llvm::Optional<std::string> getCanonicalPath(const FileEntry *F,
                                             const SourceManager &SourceMgr);

bool isRangeConsecutive(const Range &Left, const Range &Right);

format::FormatStyle getFormatStyleForFile(llvm::StringRef File,
                                          llvm::StringRef Content,
                                          llvm::vfs::FileSystem *FS);

// Cleanup and format the given replacements.
llvm::Expected<tooling::Replacements>
cleanupAndFormat(StringRef Code, const tooling::Replacements &Replaces,
                 const format::FormatStyle &Style);

} // namespace clangd
} // namespace clang
#endif
