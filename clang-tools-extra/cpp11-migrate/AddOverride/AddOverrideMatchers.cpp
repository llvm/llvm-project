//===-- AddOverride/AddOverrideMatchers.cpp - C++11 override ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definitions for matcher-generating functions
///  and a custom AST_MATCHER for identifying casts of type CK_NullTo*.
///
//===----------------------------------------------------------------------===//
#include "AddOverrideMatchers.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;
using namespace clang;

const char *MethodId = "method";

DeclarationMatcher makeCandidateForOverrideAttrMatcher() {
  return methodDecl(hasParent(recordDecl()),
                    isOverride(),
                    unless(destructorDecl())).bind(MethodId);
}

