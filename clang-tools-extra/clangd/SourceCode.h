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
#include "Protocol.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
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

// Counts the number of UTF-16 code units needed to represent a string (LSP
// specifies string lengths in UTF-16 code units).
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
} // namespace clangd
} // namespace clang
#endif
