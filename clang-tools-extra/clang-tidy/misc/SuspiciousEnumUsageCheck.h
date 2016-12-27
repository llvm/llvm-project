//===--- SuspiciousEnumUsageCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_ENUM_USAGE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_ENUM_USAGE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// The checker detects various cases when an enum is probably misused (as a
/// bitmask).
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-suspicious-enum-usage.html
class SuspiciousEnumUsageCheck : public ClangTidyCheck {
public:
  SuspiciousEnumUsageCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  void checkSuspiciousBitmaskUsage(const Expr*, const EnumDecl*);
  const bool StrictMode;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SUSPICIOUS_ENUM_USAGE_H
