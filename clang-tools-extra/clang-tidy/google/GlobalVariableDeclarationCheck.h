//===--- GlobalVariableDeclarationCheck.h - clang-tidy-----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace objc {

/// The check for Objective-C global variables and constants naming convention.
/// The declaration should follow the patterns of 'k[A-Z].*' (constants) or
/// 'g[A-Z].*' (variables).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google-objc-global-variable-declaration.html
class GlobalVariableDeclarationCheck : public ClangTidyCheck {
 public:
  GlobalVariableDeclarationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace objc
}  // namespace google
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_OBJC_GLOBAL_VARIABLE_DECLARATION_H
