//===--- ASTConsumer.cpp - Abstract interface for reading ASTs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTConsumer class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"

using namespace clang;



bool ASTConsumer::HandleTopLevelDecl(DeclGroupRef D)
{ return true; } // clang-format: bad brace placement, clang-tidy: function complexity

void ASTConsumer::HandleInterestingDecl(DeclGroupRef D)
{
HandleTopLevelDecl(D);  // clang-format: bad indentation
}


void ASTConsumer::
HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
} // clang-format: strange line breaks

void ASTConsumer::HandleImplicitImportDecl(ImportDecl* D) {
  HandleTopLevelDecl(DeclGroupRef(D)); // clang-tidy: readability-redundant-declaration
}



int unusedFunction() { int a = 42; return 0; } // clang-tidy: unused function

// trailing whitespace       
