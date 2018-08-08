//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/Tooling/Core/Replacement.h"

namespace clang {
class SourceManager;

namespace clangd {

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

/// Get the absolute file path of a given file entry.
llvm::Optional<std::string> getAbsoluteFilePath(const FileEntry *F,
                                                const SourceManager &SourceMgr);

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L);

} // namespace clangd
} // namespace clang
#endif
