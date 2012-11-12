//===-- loop-convert/LoopMatchers.h - Matchers for for loops ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of the matchers for use in migrating
// C++ for loops. The matchers are responsible for checking the general shape of
// the for loop, namely the init, condition, and increment portions.
// Further analysis will be needed to confirm that the loop is in fact
// convertible in the matcher callback.
//
//===----------------------------------------------------------------------===//
#ifndef _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOP_MATCHERS_H_
#define _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOP_MATCHERS_H_

#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace loop_migrate {

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

ast_matchers::StatementMatcher makeArrayLoopMatcher();
ast_matchers::StatementMatcher makeIteratorLoopMatcher();
ast_matchers::StatementMatcher makePseudoArrayLoopMatcher();

} //namespace loop_migrate
} //namespace clang

#endif //_LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_LOOP_MATCHERS_H_
