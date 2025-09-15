//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the FormatStringConverter class which is used to convert
/// printf format strings to C++ std::formatter format strings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FORMATSTRINGCONVERTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FORMATSTRINGCONVERTER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/FormatString.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <string>

namespace clang::tidy::utils {

/// Convert a printf-style format string to a std::formatter-style one, and
/// prepare any casts that are required to wrap the arguments to retain printf
/// compatibility. This class is expecting to work on the already-cooked format
/// string (i.e. all the escapes have been converted) so we have to convert them
/// back. This means that we might not convert them back using the same form.
class FormatStringConverter
    : public clang::analyze_format_string::FormatStringHandler {
public:
  using ConversionSpecifier = clang::analyze_format_string::ConversionSpecifier;
  using PrintfSpecifier = analyze_printf::PrintfSpecifier;

  struct Configuration {
    bool StrictMode = false;
    bool AllowTrailingNewlineRemoval = false;
  };

  FormatStringConverter(ASTContext *Context, const CallExpr *Call,
                        unsigned FormatArgOffset, Configuration Config,
                        const LangOptions &LO, SourceManager &SM,
                        Preprocessor &PP);

  bool canApply() const { return ConversionNotPossibleReason.empty(); }
  const std::string &conversionNotPossibleReason() const {
    return ConversionNotPossibleReason;
  }
  void applyFixes(DiagnosticBuilder &Diag, SourceManager &SM);
  bool usePrintNewlineFunction() const { return UsePrintNewlineFunction; }

private:
  ASTContext *Context;
  const Configuration Config;
  const bool CastMismatchedIntegerTypes;
  const Expr *const *Args;
  const unsigned NumArgs;
  unsigned ArgsOffset;
  const LangOptions &LangOpts;
  std::string ConversionNotPossibleReason;
  bool FormatStringNeededRewriting = false;
  bool UsePrintNewlineFunction = false;
  size_t PrintfFormatStringPos = 0U;
  StringRef PrintfFormatString;

  /// Lazily-created c_str() call matcher
  std::optional<clang::ast_matchers::StatementMatcher>
      StringCStrCallExprMatcher;

  const StringLiteral *FormatExpr;
  std::string StandardFormatString;

  /// Casts to be used to wrap arguments to retain printf compatibility.
  struct ArgumentFix {
    unsigned ArgIndex;
    std::string Fix;

    // We currently need this for emplace_back. Roll on C++20.
    explicit ArgumentFix(unsigned ArgIndex, std::string Fix)
        : ArgIndex(ArgIndex), Fix(std::move(Fix)) {}
  };

  std::vector<ArgumentFix> ArgFixes;
  std::vector<clang::ast_matchers::BoundNodes> ArgCStrRemovals;

  // Argument rotations to cope with the fact that std::print puts the value to
  // be formatted first and the width and precision afterwards whereas printf
  // puts the width and preicision first.
  std::vector<std::tuple<unsigned, unsigned>> ArgRotates;

  void emitAlignment(const PrintfSpecifier &FS, std::string &FormatSpec);
  void emitSign(const PrintfSpecifier &FS, std::string &FormatSpec);
  void emitAlternativeForm(const PrintfSpecifier &FS, std::string &FormatSpec);
  void emitFieldWidth(const PrintfSpecifier &FS, std::string &FormatSpec);
  void emitPrecision(const PrintfSpecifier &FS, std::string &FormatSpec);
  void emitStringArgument(unsigned ArgIndex, const Expr *Arg);
  bool emitIntegerArgument(ConversionSpecifier::Kind ArgKind, const Expr *Arg,
                           unsigned ArgIndex, std::string &FormatSpec);

  bool emitType(const PrintfSpecifier &FS, const Expr *Arg,
                std::string &FormatSpec);
  bool convertArgument(const PrintfSpecifier &FS, const Expr *Arg,
                       std::string &StandardFormatString);

  void maybeRotateArguments(const PrintfSpecifier &FS);

  bool HandlePrintfSpecifier(const PrintfSpecifier &FS,
                             const char *StartSpecifier, unsigned SpecifierLen,
                             const TargetInfo &Target) override;

  void appendFormatText(StringRef Text);
  void finalizeFormatText();
  static std::optional<StringRef>
  formatStringContainsUnreplaceableMacro(const CallExpr *CallExpr,
                                         const StringLiteral *FormatExpr,
                                         SourceManager &SM, Preprocessor &PP);
  bool conversionNotPossible(std::string Reason) {
    ConversionNotPossibleReason = std::move(Reason);
    return false;
  }
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FORMATSTRINGCONVERTER_H
