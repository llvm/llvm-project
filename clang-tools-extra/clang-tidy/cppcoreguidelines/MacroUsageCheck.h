//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Lex/MacroInfo.h"
#include <string>

namespace clang {
class MacroDirective;
namespace tidy::cppcoreguidelines {

/// Find macro usage that is considered problematic because better language
/// constructs exist for the task.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/macro-usage.html
class MacroUsageCheck : public ClangTidyCheck {
public:
  MacroUsageCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        AllowedRegexp(Options.get("AllowedRegexp", "^DEBUG_*")),
        CheckCapsOnly(Options.get("CheckCapsOnly", false)),
        IgnoreCommandLineMacros(Options.get("IgnoreCommandLineMacros", true)) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void warnMacro(const MacroDirective *MD, StringRef MacroName);
  void warnNaming(const MacroDirective *MD, StringRef MacroName);

private:
  /// A regular expression that defines how allowed macros must look like.
  std::string AllowedRegexp;
  /// Control if only the check shall only test on CAPS_ONLY macros.
  bool CheckCapsOnly;
  /// Should the macros without a valid location be diagnosed?
  bool IgnoreCommandLineMacros;
};

} // namespace tidy::cppcoreguidelines
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H
