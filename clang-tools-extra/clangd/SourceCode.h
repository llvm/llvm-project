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
#include "clang/Basic/SourceLocation.h"

namespace clang {
class SourceManager;

namespace clangd {

/// Turn a [line, column] pair into an offset in Code.
///
/// If the character value is greater than the line length, the behavior depends
/// on AllowColumnsBeyondLineLength:
///
/// - if true: default back to the end of the line
/// - if false: return an error
///
/// If the line number is greater than the number of lines in the document,
/// always return an error.
///
/// The returned value is in the range [0, Code.size()].
llvm::Expected<size_t>
positionToOffset(llvm::StringRef Code, Position P,
                 bool AllowColumnsBeyondLineLength = true);

/// Turn an offset in Code into a [line, column] pair.
/// FIXME: This should return an error if the offset is invalid.
Position offsetToPosition(llvm::StringRef Code, size_t Offset);

/// Turn a SourceLocation into a [line, column] pair.
/// FIXME: This should return an error if the location is invalid.
Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc);

// Converts a half-open clang source range to an LSP range.
// Note that clang also uses closed source ranges, which this can't handle!
Range halfOpenToRange(const SourceManager &SM, CharSourceRange R);

/// From "a::b::c", return {"a::b::", "c"}. Scope is empty if there's no
/// qualifier.
std::pair<llvm::StringRef, llvm::StringRef>
splitQualifiedName(llvm::StringRef QName);

} // namespace clangd
} // namespace clang
#endif
