//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNIQUEPTRRESETRELEASECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNIQUEPTRRESETRELEASECHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::misc {

/// Find and replace `unique_ptr::reset(release())` with `std::move()`.
///
/// Example:
///
/// \code
///   std::unique_ptr<Foo> x, y;
///   x.reset(y.release()); -> x = std::move(y);
/// \endcode
///
/// If `y` is already rvalue, `std::move()` is not added.  `x` and `y` can also
/// be `std::unique_ptr<Foo>*`.
class UniqueptrResetReleaseCheck : public ClangTidyCheck {
public:
  UniqueptrResetReleaseCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    // Only register the matchers for C++11; the functionality currently does
    // not
    // provide any benefit to other languages, despite being benign.
    return LangOpts.CPlusPlus11;
  }
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  utils::IncludeInserter Inserter;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNIQUEPTRRESETRELEASECHECK_H
