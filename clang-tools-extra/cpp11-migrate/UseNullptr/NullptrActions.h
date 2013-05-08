//===-- nullptr-convert/NullptrActions.h - Matcher callback ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the declaration of the NullptrFixer class which
///  is used as a ASTMatcher callback.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_NULLPTR_ACTIONS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_NULLPTR_ACTIONS_H

#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

// The type for user-defined macro names that behave like NULL
typedef llvm::SmallVector<llvm::StringRef, 1> UserMacroNames;

/// \brief The callback to be used for nullptr migration matchers.
///
class NullptrFixer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  NullptrFixer(clang::tooling::Replacements &Replace,
               unsigned &AcceptedChanges,
               RiskLevel);

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result);

private:
  clang::tooling::Replacements &Replace;
  unsigned &AcceptedChanges;
  UserMacroNames UserNullMacros;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_NULLPTR_ACTIONS_H
