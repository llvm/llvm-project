//===-- AddOverride/AddOverrideMatchers.h - add C++11 override -*- C++ -*-===//
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
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_MATCHERS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

/// Name to bind with matched expressions.
extern const char *MethodId;

/// \brief Create a matcher that finds member function declarations that are
/// candidates for adding the override attribute.
clang::ast_matchers::DeclarationMatcher makeCandidateForOverrideAttrMatcher();

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_MATCHERS_H
