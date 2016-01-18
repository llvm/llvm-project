//===---- CheckerHelpers.cpp - Helper functions for checkers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines several static functions for use in checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/AST/Expr.h"

// Recursively find any substatements containing macros
bool clang::ento::containsMacro(const Stmt *S) {
  if (S->getLocStart().isMacroID())
    return true;

  if (S->getLocEnd().isMacroID())
    return true;

  for (const Stmt *Child : S->children())
    if (Child && containsMacro(Child))
      return true;

  return false;
}

// Recursively find any substatements containing enum constants
bool clang::ento::containsEnum(const Stmt *S) {
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S);

  if (DR && isa<EnumConstantDecl>(DR->getDecl()))
    return true;

  for (const Stmt *Child : S->children())
    if (Child && containsEnum(Child))
      return true;

  return false;
}

// Recursively find any substatements containing static vars
bool clang::ento::containsStaticLocal(const Stmt *S) {
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S);

  if (DR)
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl()))
      if (VD->isStaticLocal())
        return true;

  for (const Stmt *Child : S->children())
    if (Child && containsStaticLocal(Child))
      return true;

  return false;
}

// Recursively find any substatements containing __builtin_offsetof
bool clang::ento::containsBuiltinOffsetOf(const Stmt *S) {
  if (isa<OffsetOfExpr>(S))
    return true;

  for (const Stmt *Child : S->children())
    if (Child && containsBuiltinOffsetOf(Child))
      return true;

  return false;
}
