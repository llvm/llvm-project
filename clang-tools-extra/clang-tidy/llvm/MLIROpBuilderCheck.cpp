//===--- MLIROpBuilderCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLIROpBuilderCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LLVM.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/Support/Error.h"

namespace clang::tidy::llvm_check {
namespace {

using namespace ::clang::ast_matchers;
using namespace ::clang::transformer;

class TypeAsWrittenStencil : public StencilInterface {
public:
  explicit TypeAsWrittenStencil(std::string S) : Id(std::move(S)) {}
  std::string toString() const override {
    return (llvm::Twine("TypeAsWritten(\"") + Id + "\")").str();
  }

  llvm::Error eval(const MatchFinder::MatchResult &match,
                   std::string *result) const override {
    llvm::Expected<CharSourceRange> n = node(Id)(match);
    if (!n)
      return n.takeError();
    const SourceRange SrcRange = n->getAsRange();
    if (SrcRange.isInvalid()) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "SrcRange is invalid");
    }
    const CharSourceRange Range = n->getTokenRange(SrcRange);
    auto NextToken = [&](std::optional<Token> Token) {
      if (!Token)
        return Token;
      return clang::Lexer::findNextToken(Token->getLocation(),
                                         *match.SourceManager,
                                         match.Context->getLangOpts());
    };
    std::optional<Token> LessToken = clang::Lexer::findNextToken(
        Range.getBegin(), *match.SourceManager, match.Context->getLangOpts());
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
    *result += clang::tooling::getText(
        CharSourceRange::getTokenRange(LessToken->getEndLoc(),
                                       EndToken->getLastLoc()),
        *match.Context);
    return llvm::Error::success();
  }
  std::string Id;
};

Stencil typeAsWritten(StringRef Id) {
  // Using this instead of `describe` so that we get the exact same spelling.
  return std::make_shared<TypeAsWrittenStencil>(std::string(Id));
}

RewriteRuleWith<std::string> mlirOpBuilderCheckRule() {
  return makeRule(
      cxxMemberCallExpr(
          on(expr(hasType(
                      cxxRecordDecl(isSameOrDerivedFrom("::mlir::OpBuilder"))))
                 .bind("builder")),
          callee(cxxMethodDecl(hasTemplateArgument(0, templateArgument()))),
          callee(cxxMethodDecl(hasName("create"))))
          .bind("call"),
      changeTo(cat(typeAsWritten("call"), "::create(", node("builder"), ", ",
                   callArgs("call"), ")")),
      cat("use 'OpType::create(builder, ...)' instead of "
          "'builder.create<OpType>(...)'"));
}
} // namespace

MlirOpBuilderCheck::MlirOpBuilderCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : TransformerClangTidyCheck(mlirOpBuilderCheckRule(), Name, Context) {}

} // namespace clang::tidy::llvm_check
