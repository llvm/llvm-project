//===-- AddOverride/AddOverrideActions.h - add C++11 override --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the declaration of the AddOverrideFixer class
///  which is used as a ASTMatcher callback.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_ACTIONS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_ACTIONS_H

#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"

/// \brief The callback to be used for add-override migration matchers.
///
class AddOverrideFixer : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  AddOverrideFixer(clang::tooling::Replacements &Replace,
                   unsigned &AcceptedChanges) :
    Replace(Replace), AcceptedChanges(AcceptedChanges) {}

  /// \brief Entry point to the callback called when matches are made.
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result);

private:
  clang::tooling::Replacements &Replace;
  unsigned &AcceptedChanges;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_ADD_OVERRIDE_ACTIONS_H
