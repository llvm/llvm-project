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
size_t positionToOffset(llvm::StringRef Code, Position P);

/// Turn an offset in Code into a [line, column] pair.
Position offsetToPosition(llvm::StringRef Code, size_t Offset);

/// Turn a SourceLocation into a [line, column] pair.
Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc);

} // namespace clangd
} // namespace clang
#endif
