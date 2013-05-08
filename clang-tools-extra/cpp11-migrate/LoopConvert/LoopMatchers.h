//===-- LoopConvert/LoopMatchers.h - Matchers for for loops -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains declarations of the matchers for use in migrating
/// C++ for loops. The matchers are responsible for checking the general shape
/// of the for loop, namely the init, condition, and increment portions.
/// Further analysis will be needed to confirm that the loop is in fact
/// convertible in the matcher callback.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_MATCHERS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

// Constants used for matcher name bindings
extern const char LoopName[];
extern const char ConditionBoundName[];
extern const char ConditionVarName[];
extern const char ConditionEndVarName[];
extern const char IncrementVarName[];
extern const char InitVarName[];
extern const char EndExprName[];
extern const char EndCallName[];
extern const char EndVarName[];
extern const char DerefByValueResultName[];

clang::ast_matchers::StatementMatcher makeArrayLoopMatcher();
clang::ast_matchers::StatementMatcher makeIteratorLoopMatcher();
clang::ast_matchers::StatementMatcher makePseudoArrayLoopMatcher();

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_LOOP_MATCHERS_H
