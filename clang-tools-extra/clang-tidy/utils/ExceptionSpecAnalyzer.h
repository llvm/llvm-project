//===--- ExceptionSpecAnalyzer.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_SPEC_ANALYZER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_SPEC_ANALYZER_H

#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/DenseMap.h"

namespace clang::tidy::utils {

/// This class analysis if a `FunctionDecl` has been declared implicitly through
/// defaulting or explicitly as throwing or not and evaluates noexcept
/// expressions if needed. Unlike the `ExceptionAnalyzer` however it can't tell
/// you if the function will actually throw an exception or not.
class ExceptionSpecAnalyzer {
public:
  enum class State {
    Throwing,    ///< This function has been declared as possibly throwing.
    NotThrowing, ///< This function has been declared as not throwing.
    Unknown, ///< We're unable to tell if this function is declared as throwing
             ///< or not.
  };

  ExceptionSpecAnalyzer() = default;

  State analyze(const FunctionDecl *FuncDecl);

private:
  enum class DefaultableMemberKind {
    DefaultConstructor,
    CopyConstructor,
    MoveConstructor,
    CopyAssignment,
    MoveAssignment,
    Destructor,

    CompareEqual,
    CompareNotEqual,
    CompareThreeWay,
    CompareRelational,

    None,
  };

  State analyzeImpl(const FunctionDecl *FuncDecl);

  State analyzeUnresolvedOrDefaulted(const CXXMethodDecl *MethodDecl,
                                     const FunctionProtoType *FuncProto);

  State analyzeFieldDecl(const FieldDecl *FDecl, DefaultableMemberKind Kind);

  State analyzeBase(const CXXBaseSpecifier &Base, DefaultableMemberKind Kind);

  enum class SkipMethods : bool {
    Yes = true,
    No = false,
  };

  State analyzeRecord(const CXXRecordDecl *RecDecl, DefaultableMemberKind Kind,
                      SkipMethods SkipMethods = SkipMethods::No);

  static State analyzeFunctionEST(const FunctionDecl *FuncDecl,
                                  const FunctionProtoType *FuncProto);

  static bool hasTrivialMemberKind(const CXXRecordDecl *RecDecl,
                                   DefaultableMemberKind Kind);

  static bool isConstructor(DefaultableMemberKind Kind);

  static bool isSpecialMember(DefaultableMemberKind Kind);

  static bool isComparison(DefaultableMemberKind Kind);

  static DefaultableMemberKind
  getDefaultableMemberKind(const FunctionDecl *FuncDecl);

  llvm::DenseMap<const FunctionDecl *, State> FunctionCache{32u};
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_SPEC_ANALYZER_H
