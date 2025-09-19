//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_STRUCTPACKALIGNCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_STRUCTPACKALIGNCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::altera {

/// Finds structs that are inefficiently packed or aligned, and recommends
/// packing and/or aligning of said structs as needed.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/altera/struct-pack-align.html
class StructPackAlignCheck : public ClangTidyCheck {
public:
  StructPackAlignCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        MaxConfiguredAlignment(Options.get("MaxConfiguredAlignment", 128)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const unsigned MaxConfiguredAlignment;
  CharUnits computeRecommendedAlignment(CharUnits MinByteSize) const;
};

} // namespace clang::tidy::altera

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_STRUCTPACKALIGNCHECK_H
