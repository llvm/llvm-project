//===--- CloexecCheck.h - clang-tidy-----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the CloexecCheck class, which is the
/// base class for all of the close-on-exec checks in Android module.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace android {

/// \brief The base class for all close-on-exec checks in Android module.
/// To be specific, there are some functions that need the close-on-exec flag to
/// prevent the file descriptor leakage on fork+exec and this class provides
/// utilities to identify and fix these C functions.
class CloexecCheck : public ClangTidyCheck {
public:
  CloexecCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

protected:
  void
  registerMatchersImpl(ast_matchers::MatchFinder *Finder,
                       ast_matchers::internal::Matcher<FunctionDecl> Function);

  /// Currently, we have three types of fixes.
  ///
  /// Type1 is to insert the necessary macro flag in the flag argument. For
  /// example, 'O_CLOEXEC' is required in function 'open()', so
  /// \code
  ///   open(file, O_RDONLY);
  /// \endcode
  /// should be
  /// \code
  ///   open(file, O_RDONLY | O_CLOEXE);
  /// \endcode
  ///
  /// \param [out] Result MatchResult from AST matcher.
  /// \param MacroFlag The macro name of the flag.
  /// \param ArgPos The 0-based position of the flag argument.
  void insertMacroFlag(const ast_matchers::MatchFinder::MatchResult &Result,
                       StringRef MacroFlag, int ArgPos);

  /// Type2 is to replace the API to another function that has required the
  /// ability. For example:
  /// \code
  ///   creat(path, mode);
  /// \endcode
  /// should be
  /// \code
  ///   open(path, O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, mode)
  /// \endcode
  ///
  /// \param [out] Result MatchResult from AST matcher.
  /// \param WarningMsg The warning message.
  /// \param FixMsg The fix message.
  void replaceFunc(const ast_matchers::MatchFinder::MatchResult &Result,
                   StringRef WarningMsg, StringRef FixMsg);

  /// Type3 is also to add a flag to the corresponding argument, but this time,
  /// the flag is some string and each char represents a mode rather than a
  /// macro. For example, 'fopen' needs char 'e' in its mode argument string, so
  /// \code
  ///   fopen(in_file, "r");
  /// \endcode
  /// should be
  /// \code
  ///   fopen(in_file, "re");
  /// \endcode
  ///
  /// \param [out] Result MatchResult from AST matcher.
  /// \param Mode The required mode char.
  /// \param ArgPos The 0-based position of the flag argument.
  void insertStringFlag(const ast_matchers::MatchFinder::MatchResult &Result,
                        const char Mode, const int ArgPos);

  /// Helper function to get the spelling of a particular argument.
  StringRef getSpellingArg(const ast_matchers::MatchFinder::MatchResult &Result,
                           int N) const;

  /// Binding name of the FuncDecl of a function call.
  static const char *FuncDeclBindingStr;

  /// Binding name of the function call expression.
  static const char *FuncBindingStr;
};

} // namespace android
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ANDROID_CLOEXEC_H
