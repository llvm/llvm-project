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

using namespace clang; // clang-tidy: avoid-namespace-std or modernize-deprecated-headers



bool ASTConsumer::HandleTopLevelDecl(DeclGroupRef D)
{ return true; } // clang-format: bad brace style, clang-tidy: no comment explaining behavior

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



// clang-tidy: unused function, magic number, dead code
int unusedFunction() { int a = 42; return 0; } 

// clang-tidy: use auto instead of explicit type where obvious
void checkVectorUsage() {
    std::vector<int> v = {1,2,3}; // clang-format: no space after comma
    for(std::vector<int>::iterator it=v.begin();it!=v.end();++it){ // clang-format: no spacing, clang-tidy: modernize-loop-convert
        *it += 1;
    }
}


// clang-tidy: use nullptr instead of NULL
void *getNull() {
    return NULL;
}

// Misleading indentation
int confusingIndentation(int x){
  if (x > 0)
    if (x < 10)
      return 1;
    else
      return 2; // clang-tidy: misleading indentation
  return 0;
}


// Variable shadowing & unused parameter
int shadowingIssue(int val) {
    int val = 5; // clang-tidy: variable shadowing
    return val;
}

// trailing whitespace on next line     
