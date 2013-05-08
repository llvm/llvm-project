//===-- UseAutoMatchers.h - Matchers for use-auto transform ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the declarations for matcher-generating functions
///  and names for bound nodes found by AST matchers.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_AUTO_MATCHERS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_AUTO_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

extern const char *IteratorDeclId;
extern const char *DeclWithNewId;
extern const char *NewExprId;

/// \brief Create a matcher that matches variable declarations where the type
/// is an iterator for an std container and has an explicit initializer of the
/// same type.
clang::ast_matchers::DeclarationMatcher makeIteratorDeclMatcher();

/// \brief Create a matcher that matches variable declarations that are
/// initialized by a C++ new expression.
clang::ast_matchers::DeclarationMatcher makeDeclWithNewMatcher();

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_AUTO_MATCHERS_H
