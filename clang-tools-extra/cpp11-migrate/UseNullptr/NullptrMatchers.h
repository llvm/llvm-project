//===-- nullptr-convert/Matchers.h - Matchers for null casts ---*- C++ -*--===//
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
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

// Names to bind with matched expressions.
extern const char *CastSequence;

/// \brief Create a matcher that finds implicit casts as well as the head of a
/// sequence of zero or more nested explicit casts that have an implicit cast
/// to null within.
/// Finding sequences of explict casts is necessary so that an entire sequence
/// can be replaced instead of just the inner-most implicit cast.
clang::ast_matchers::StatementMatcher makeCastSequenceMatcher();

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H
