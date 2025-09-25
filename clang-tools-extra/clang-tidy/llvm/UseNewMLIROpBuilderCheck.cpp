//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseNewMLIROpBuilderCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang::tidy::llvm_check {
namespace {

using namespace ::clang::ast_matchers;
using namespace ::clang::transformer;

EditGenerator rewrite(RangeSelector Call, RangeSelector Builder,
                      RangeSelector CallArgs) {
  // This is using an EditGenerator rather than ASTEdit as we want to warn even
  // if in macro.
  return [Call = std::move(Call), Builder = std::move(Builder),
          CallArgs =
              std::move(CallArgs)](const MatchFinder::MatchResult &Result)
             -> Expected<SmallVector<transformer::Edit, 1>> {
    Expected<CharSourceRange> CallRange = Call(Result);
    if (!CallRange)
      return CallRange.takeError();
    SourceManager &SM = *Result.SourceManager;
    const LangOptions &LangOpts = Result.Context->getLangOpts();
    SourceLocation Begin = CallRange->getBegin();

    // This will result in just a warning and no edit.
    bool InMacro = CallRange->getBegin().isMacroID();
    if (InMacro) {
      while (SM.isMacroArgExpansion(Begin))
        Begin = SM.getImmediateExpansionRange(Begin).getBegin();
      Edit WarnOnly;
      WarnOnly.Kind = EditKind::Range;
      WarnOnly.Range = CharSourceRange::getCharRange(Begin, Begin);
      return SmallVector<Edit, 1>({WarnOnly});
    }

    // This will try to extract the template argument as written so that the
    // rewritten code looks closest to original.
    auto NextToken = [&](std::optional<Token> CurrentToken) {
      if (!CurrentToken)
        return CurrentToken;
      if (CurrentToken->getEndLoc() >= CallRange->getEnd())
        return std::optional<Token>();
      return clang::Lexer::findNextToken(CurrentToken->getLocation(), SM,
                                         LangOpts);
    };
    std::optional<Token> LessToken =
        clang::Lexer::findNextToken(Begin, SM, LangOpts);
    while (LessToken && LessToken->getKind() != clang::tok::less) {
      LessToken = NextToken(LessToken);
    }
    if (!LessToken) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "missing '<' token");
    }
    std::optional<Token> EndToken = NextToken(LessToken);
    for (std::optional<Token> GreaterToken = NextToken(EndToken);
         GreaterToken && GreaterToken->getKind() != clang::tok::greater;
         GreaterToken = NextToken(GreaterToken)) {
      EndToken = GreaterToken;
    }
    if (!EndToken) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "missing '>' token");
    }

    Expected<CharSourceRange> BuilderRange = Builder(Result);
    if (!BuilderRange)
      return BuilderRange.takeError();
    Expected<CharSourceRange> CallArgsRange = CallArgs(Result);
    if (!CallArgsRange)
      return CallArgsRange.takeError();

    // Helper for concatting below.
    auto GetText = [&](const CharSourceRange &Range) {
      return clang::Lexer::getSourceText(Range, SM, LangOpts);
    };

    Edit Replace;
    Replace.Kind = EditKind::Range;
    Replace.Range = *CallRange;
    std::string CallArgsStr;
    // Only emit args if there are any.
    if (auto CallArgsText = GetText(*CallArgsRange).ltrim();
        !CallArgsText.rtrim().empty()) {
      CallArgsStr = llvm::formatv(", {}", CallArgsText);
    }
    Replace.Replacement =
        llvm::formatv("{}::create({}{})",
                      GetText(CharSourceRange::getTokenRange(
                          LessToken->getEndLoc(), EndToken->getLastLoc())),
                      GetText(*BuilderRange), CallArgsStr);

    return SmallVector<Edit, 1>({Replace});
  };
}

RewriteRuleWith<std::string> useNewMlirOpBuilderCheckRule() {
  Stencil message = cat("use 'OpType::create(builder, ...)' instead of "
                        "'builder.create<OpType>(...)'");
  // Match a create call on an OpBuilder.
  ast_matchers::internal::Matcher<Stmt> base =
      cxxMemberCallExpr(
          on(expr(hasType(
                      cxxRecordDecl(isSameOrDerivedFrom("::mlir::OpBuilder"))))
                 .bind("builder")),
          callee(cxxMethodDecl(hasTemplateArgument(0, templateArgument()))),
          callee(cxxMethodDecl(hasName("create"))))
          .bind("call");
  return applyFirst(
      //  Attempt rewrite given an lvalue builder, else just warn.
      {makeRule(cxxMemberCallExpr(unless(on(cxxTemporaryObjectExpr())), base),
                rewrite(node("call"), node("builder"), callArgs("call")),
                message),
       makeRule(base, noopEdit(node("call")), message)});
}
} // namespace

UseNewMlirOpBuilderCheck::UseNewMlirOpBuilderCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : TransformerClangTidyCheck(useNewMlirOpBuilderCheckRule(), Name, Context) {
}

} // namespace clang::tidy::llvm_check
