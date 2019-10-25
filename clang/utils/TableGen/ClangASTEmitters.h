//=== ClangASTEmitters.h - Definitions for AST node emitters ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_AST_EMITTERS
#define CLANG_AST_EMITTERS

#include "llvm/TableGen/Record.h"
#include "llvm/ADT/STLExtras.h"

// These are spellings in the tblgen files.

// The field name for the base-node property.
// Fortunately, this is common across all the hierarchies.
#define BaseFieldName "Base"

// Comment node hierarchy.
#define CommentNodeClassName "CommentNode"

// Decl node hierarchy.
#define DeclNodeClassName "DeclNode"
#define DeclContextNodeClassName "DeclContext"

// Stmt node hierarchy.
#define StmtNodeClassName "StmtNode"

// Type node hierarchy.
#define TypeNodeClassName "TypeNode"
#define AlwaysDependentClassName "AlwaysDependent"
#define NeverCanonicalClassName "NeverCanonical"
#define NeverCanonicalUnlessDependentClassName "NeverCanonicalUnlessDependent"
#define LeafTypeClassName "LeafType"
#define AbstractFieldName "Abstract"

#endif
