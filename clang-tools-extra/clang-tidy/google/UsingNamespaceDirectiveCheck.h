//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google::build {

/// Finds using namespace directives.
///
/// https://google.github.io/styleguide/cppguide.html#Namespaces
///
/// The check implements the following rule of the Google C++ Style Guide:
///
///   You may not use a using-directive to make all names from a namespace
///   available.
///
///   \code
///     // Forbidden -- This pollutes the namespace.
///     using namespace foo;
///   \endcode
///
/// Corresponding cpplint.py check name: `build/namespaces`.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/google/build-using-namespace.html
class UsingNamespaceDirectiveCheck : public ClangTidyCheck {
public:
  UsingNamespaceDirectiveCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static bool isStdLiteralsNamespace(const NamespaceDecl *NS);
};

} // namespace clang::tidy::google::build

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H
