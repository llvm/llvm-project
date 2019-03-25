//===--- UnnamedNamespaceInHeaderCheck.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/HeaderFileExtensionsUtils.h"

namespace clang {
namespace tidy {
namespace google {
namespace build {

/// Finds anonymous namespaces in headers.
///
/// The check supports these options:
///   - `HeaderFileExtensions`: a comma-separated list of filename extensions of
///     header files (The filename extensions should not contain "." prefix).
///     "h,hh,hpp,hxx" by default.
///     For extension-less header files, using an empty string or leaving an
///     empty string between "," if there are other filename extensions.
///
/// https://google.github.io/styleguide/cppguide.html#Namespaces
///
/// Corresponding cpplint.py check name: 'build/namespaces'.
class UnnamedNamespaceInHeaderCheck : public ClangTidyCheck {
public:
  UnnamedNamespaceInHeaderCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::string RawStringHeaderFileExtensions;
  utils::HeaderFileExtensionsSet HeaderFileExtensions;
};

} // namespace build
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H
