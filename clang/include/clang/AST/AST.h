//===--- AST.h - "Umbrella" header for AST library --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to the AST classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_AST_H
#define LLVM_CLANG_AST_AST_H

// This header exports all AST interfaces.
#include "clang/AST/ASTContext.h"   // IWYU pragma: export
#include "clang/AST/Decl.h"         // IWYU pragma: export
#include "clang/AST/DeclCXX.h"      // IWYU pragma: export
#include "clang/AST/DeclObjC.h"     // IWYU pragma: export
#include "clang/AST/DeclTemplate.h" // IWYU pragma: export
#include "clang/AST/Expr.h"         // IWYU pragma: export
#include "clang/AST/ExprObjC.h"     // IWYU pragma: export
#include "clang/AST/StmtVisitor.h"  // IWYU pragma: export
#include "clang/AST/Type.h"         // IWYU pragma: export

#endif
