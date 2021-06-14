//===--- NarrowingConversionsCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NARROWING_CONVERSIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NARROWING_CONVERSIONS_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Checks for narrowing conversions, e.g:
///   int i = 0;
///   i += 0.1;
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-narrowing-conversions.html
class NarrowingConversionsCheck : public ClangTidyCheck {
public:
  NarrowingConversionsCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void diagNarrowType(SourceLocation SourceLoc, const Expr &Lhs,
                      const Expr &Rhs);

  void diagNarrowTypeToSignedInt(SourceLocation SourceLoc, const Expr &Lhs,
                                 const Expr &Rhs);

  void diagNarrowIntegerConstant(SourceLocation SourceLoc, const Expr &Lhs,
                                 const Expr &Rhs, const llvm::APSInt &Value);

  void diagNarrowIntegerConstantToSignedInt(SourceLocation SourceLoc,
                                            const Expr &Lhs, const Expr &Rhs,
                                            const llvm::APSInt &Value,
                                            const uint64_t HexBits);

  void diagNarrowConstant(SourceLocation SourceLoc, const Expr &Lhs,
                          const Expr &Rhs);

  void diagConstantCast(SourceLocation SourceLoc, const Expr &Lhs,
                        const Expr &Rhs);

  void diagNarrowTypeOrConstant(const ASTContext &Context,
                                SourceLocation SourceLoc, const Expr &Lhs,
                                const Expr &Rhs);

  void handleIntegralCast(const ASTContext &Context, SourceLocation SourceLoc,
                          const Expr &Lhs, const Expr &Rhs);

  void handleIntegralToBoolean(const ASTContext &Context,
                               SourceLocation SourceLoc, const Expr &Lhs,
                               const Expr &Rhs);

  void handleIntegralToFloating(const ASTContext &Context,
                                SourceLocation SourceLoc, const Expr &Lhs,
                                const Expr &Rhs);

  void handleFloatingToIntegral(const ASTContext &Context,
                                SourceLocation SourceLoc, const Expr &Lhs,
                                const Expr &Rhs);

  void handleFloatingToBoolean(const ASTContext &Context,
                               SourceLocation SourceLoc, const Expr &Lhs,
                               const Expr &Rhs);

  void handleBooleanToSignedIntegral(const ASTContext &Context,
                                     SourceLocation SourceLoc, const Expr &Lhs,
                                     const Expr &Rhs);

  void handleFloatingCast(const ASTContext &Context, SourceLocation SourceLoc,
                          const Expr &Lhs, const Expr &Rhs);

  void handleBinaryOperator(const ASTContext &Context, SourceLocation SourceLoc,
                            const Expr &Lhs, const Expr &Rhs);

  bool handleConditionalOperator(const ASTContext &Context, const Expr &Lhs,
                                 const Expr &Rhs);

  void handleImplicitCast(const ASTContext &Context,
                          const ImplicitCastExpr &Cast);

  void handleBinaryOperator(const ASTContext &Context,
                            const BinaryOperator &Op);

  bool isWarningInhibitedByEquivalentSize(const ASTContext &Context,
                                          const BuiltinType &FromType,
                                          const BuiltinType &ToType) const;

  const bool WarnOnIntegerNarrowingConversion;
  const bool WarnOnFloatingPointNarrowingConversion;
  const bool WarnWithinTemplateInstantiation;
  const bool WarnOnEquivalentBitWidth;
  const std::string IgnoreConversionFromTypes;
  const bool PedanticMode;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_NARROWING_CONVERSIONS_H
