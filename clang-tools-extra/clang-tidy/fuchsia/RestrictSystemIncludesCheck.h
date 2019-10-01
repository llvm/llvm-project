//===--- RestrictSystemIncludesCheck.h - clang-tidy---------- ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_RESTRICTINCLUDESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_RESTRICTINCLUDESSCHECK_H

#include "../ClangTidyCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "../utils/OptionsUtils.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Checks for allowed includes and suggests removal of any others. If no
/// includes are specified, the check will exit without issuing any warnings.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-restrict-system-includes.html
class RestrictSystemIncludesCheck : public ClangTidyCheck {
public:
  RestrictSystemIncludesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        AllowedIncludes(Options.get("Includes", "*")),
        AllowedIncludesGlobList(AllowedIncludes) {}

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool contains(StringRef FileName) {
    return AllowedIncludesGlobList.contains(FileName);
  }

private:
  std::string AllowedIncludes;
  GlobList AllowedIncludesGlobList;
};

} // namespace fuchsia
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_RESTRICTINCLUDESSCHECK_H
