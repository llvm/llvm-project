//===--- UsingNamespaceDirectiveCheck.h - clang-tidy ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace google {
namespace build {

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
class UsingNamespaceDirectiveCheck : public ClangTidyCheck {
public:
  UsingNamespaceDirectiveCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  static bool isStdLiteralsNamespace(const NamespaceDecl *NS);
};

} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_USINGNAMESPACEDIRECTIVECHECK_H
