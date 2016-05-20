//===--- GlobalNamesInHeadersCheck.h - clang-tidy ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBALNAMESINHEADERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBALNAMESINHEADERSCHECK_H

#include "../ClangTidy.h"
#include "../utils/HeaderFileExtensionsUtils.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

/// Flag global namespace pollution in header files.
/// Right now it only triggers on using declarations and directives.
///
/// The check supports these options:
///   - `HeaderFileExtensions`: a comma-separated list of filename extensions
///     of header files (the filename extensions should not contain "." prefix).
///     "h" by default.
///     For extension-less header files, using an empty string or leaving an
///     empty string between "," if there are other filename extensions.
class GlobalNamesInHeadersCheck : public ClangTidyCheck {
public:
  GlobalNamesInHeadersCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::string RawStringHeaderFileExtensions;
  utils::HeaderFileExtensionsSet HeaderFileExtensions;
};

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_GLOBALNAMESINHEADERSCHECK_H
