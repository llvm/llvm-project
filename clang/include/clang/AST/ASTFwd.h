//===--- ASTFwd.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//
///
/// \file
/// Forward declaration of all AST node types.
///
//===-------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTFWD_H
#define LLVM_CLANG_AST_ASTFWD_H

namespace clang {

class Decl;
#define DECL(DERIVED, BASE) class DERIVED##Decl;
#include "clang/AST/DeclNodes.inc"
class Stmt;
#define STMT(DERIVED, BASE) class DERIVED;
#include "clang/AST/StmtNodes.inc"
class Type;
#define TYPE(DERIVED, BASE) class DERIVED##Type;
#include "clang/AST/TypeNodes.def"
class CXXCtorInitializer;

} // end namespace clang

#endif
