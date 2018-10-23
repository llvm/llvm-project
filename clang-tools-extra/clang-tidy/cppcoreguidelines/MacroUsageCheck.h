//===--- MacroUsageCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H

#include "../ClangTidy.h"
#include "clang/Lex/Preprocessor.h"
#include <string>

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Find macro usage that is considered problematic because better language
/// constructs exist for the task.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-macro-usage.html
class MacroUsageCheck : public ClangTidyCheck {
public:
  MacroUsageCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        AllowedRegexp(Options.get("AllowedRegexp", "^DEBUG_*")),
        CheckCapsOnly(Options.get("CheckCapsOnly", 0)) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerPPCallbacks(CompilerInstance &Compiler) override;
  void warnMacro(const MacroDirective *MD);
  void warnNaming(const MacroDirective *MD);

private:
  /// A regular expression that defines how allowed macros must look like.
  std::string AllowedRegexp;
  /// Control if only the check shall only test on CAPS_ONLY macros.
  bool CheckCapsOnly;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MACROUSAGECHECK_H
