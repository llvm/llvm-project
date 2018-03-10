//===--- AST.h - Utility AST functions  -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Various code that examines C++ source code using AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H_

#include "clang/Basic/SourceLocation.h"

namespace clang {
class SourceManager;
class Decl;

namespace clangd {

/// Find the identifier source location of the given D.
///
/// The returned location is usually the spelling location where the name of the
/// decl occurs in the code.
SourceLocation findNameLoc(const clang::Decl* D);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_AST_H_
