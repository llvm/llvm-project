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

namespace clang {
namespace clangd {

/// Turn a [line, column] pair into an offset in Code.
size_t positionToOffset(llvm::StringRef Code, Position P);

/// Turn an offset in Code into a [line, column] pair.
Position offsetToPosition(llvm::StringRef Code, size_t Offset);

} // namespace clangd
} // namespace clang
#endif
