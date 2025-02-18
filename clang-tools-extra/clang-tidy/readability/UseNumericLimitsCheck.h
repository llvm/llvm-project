//===--- UseNumericLimitsCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USENUMERICLIMITSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USENUMERICLIMITSCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include <vector>

namespace clang::tidy::readability {

/// Replaces certain integer literals with equivalent calls to
/// ``std::numeric_limits``.
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/use-numeric-limits.html
class UseNumericLimitsCheck : public ClangTidyCheck {
public:
  UseNumericLimitsCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::vector<std::tuple<int64_t, std::string>> SignedConstants;
  const std::vector<std::tuple<uint64_t, std::string>> UnsignedConstants;
  utils::IncludeInserter Inserter;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USENUMERICLIMITSCHECK_H
