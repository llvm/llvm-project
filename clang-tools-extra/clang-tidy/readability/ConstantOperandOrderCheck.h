//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTANTOPERANDORDERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTANTOPERANDORDERCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <string>
#include <vector>

namespace clang::tidy::readability {
/// Finds binary expressions where the constant operand appears on the
/// non-preferred side (configurable) and offers a fix-it that swaps operands
/// (and inverts the operator for asymmetric operators like `<` / `>`).
/// Options
///  - BinaryOperators: comma-separated list of operators to check
///      (default: "==,!=,<,<=,>,>=")
///  - PreferredConstantSide: "Left" or "Right" (default: "Right")
class ConstantOperandOrderCheck : public ClangTidyCheck {
public:
  ConstantOperandOrderCheck(StringRef Name, ClangTidyContext *Context);
  ~ConstantOperandOrderCheck() override;

  // Persist options so they show up in config files.
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    // This check is primarily for C/C++, keep it limited to C++ for now.
    return LangOpts.CPlusPlus;
  }

private:
  // Option keys (used when storing & reading options)
  static constexpr llvm::StringLiteral BinaryOperatorsOption =
      "BinaryOperators";
  static constexpr llvm::StringLiteral PreferredSideOption =
      "PreferredConstantSide";

  // Runtime values, populated from Options in the constructor (or storeOptions)
  std::string PreferredSide;          // "Left" or "Right"
  std::vector<std::string> Operators; // list of operator names, e.g. "=="

  // Implementation helpers live in the .cpp file.
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTANTOPERANDORDERCHECK_H
