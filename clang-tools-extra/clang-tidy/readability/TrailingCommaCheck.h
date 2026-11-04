//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_TRAILINGCOMMACHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_TRAILINGCOMMACHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Checks for presence or absence of trailing commas in enum definitions
/// and initializer lists.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/trailing-comma.html
class TrailingCommaCheck : public ClangTidyCheck {
public:
  TrailingCommaCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus || LangOpts.C99;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  enum class CommaPolicyKind { Append, Remove, Ignore };

private:
  const CommaPolicyKind SingleLineCommaPolicy;
  const CommaPolicyKind MultiLineCommaPolicy;

  void checkEnumDecl(const EnumDecl *Enum,
                     const ast_matchers::MatchFinder::MatchResult &Result);
  void checkInitListExpr(const InitListExpr *InitList,
                         const ast_matchers::MatchFinder::MatchResult &Result);

  // Values correspond to %select{initializer list|enum}0 indices
  enum DiagKind { InitList = 0, Enum = 1 };
  void emitDiag(SourceLocation LastLoc, std::optional<Token> Token,
                DiagKind Kind,
                const ast_matchers::MatchFinder::MatchResult &Result,
                CommaPolicyKind Policy);
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_TRAILINGCOMMACHECK_H
