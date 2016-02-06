//===--- UnnamedNamespaceInHeaderCheck.h - clang-tidy -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H

#include "../ClangTidy.h"
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
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml?showone=Namespaces#Namespaces
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
