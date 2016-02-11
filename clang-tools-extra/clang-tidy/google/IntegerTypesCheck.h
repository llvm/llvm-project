//===--- IntegerTypesCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H

#include "../ClangTidy.h"

#include <memory>

namespace clang {

class IdentifierTable;

namespace tidy {
namespace google {
namespace runtime {

/// Finds uses of `short`, `long` and `long long` and suggest replacing them
/// with `u?intXX(_t)?`.
///
/// Correspondig cpplint.py check: 'runtime/int'.
class IntegerTypesCheck : public ClangTidyCheck {
public:
  IntegerTypesCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

private:
  const std::string UnsignedTypePrefix;
  const std::string SignedTypePrefix;
  const std::string TypeSuffix;

  std::unique_ptr<IdentifierTable> IdentTable;
};

} // namespace runtime
} // namespace google
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
