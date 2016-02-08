//===--- IncorrectRoundings.h - clang-tidy ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCORRECTROUNDINGS_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCORRECTROUNDINGS_H_

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// \brief Checks the usage of patterns known to produce incorrect rounding.
/// Programmers often use
///   (int)(double_expression + 0.5)
/// to round the double expression to an integer. The problem with this
///  1. It is unnecessarily slow.
///  2. It is incorrect. The number 0.499999975 (smallest representable float
///     number below 0.5) rounds to 1.0. Even worse behavior for negative
///     numbers where both -0.5f and -1.4f both round to 0.0.
class IncorrectRoundings : public ClangTidyCheck {
public:
  IncorrectRoundings(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_INCORRECTROUNDINGS_H_
