//===--- MLIROpBuilderCheck.cpp - clang-tidy -------------------------------===//
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

using namespace ::clang::ast_matchers; // NOLINT: Too many names.
using namespace ::clang::transformer;  // NOLINT: Too many names.

class TypeAsWrittenStencil : public StencilInterface {
public:
  explicit TypeAsWrittenStencil(std::string S) : id(std::move(S)) {}
  std::string toString() const override {
    return (llvm::Twine("TypeAsWritten(\"") + id + "\")").str();
  }

  llvm::Error eval(const MatchFinder::MatchResult &match,
                   std::string *result) const override {
    auto n = node(id)(match);
    if (!n)
      return n.takeError();
    auto srcRange = n->getAsRange();
    if (srcRange.isInvalid()) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "srcRange is invalid");
    }
    auto range = n->getTokenRange(srcRange);
    auto nextToken = [&](std::optional<Token> token) {
      if (!token)
        return token;
      return clang::Lexer::findNextToken(token->getLocation(),
                                         *match.SourceManager,
                                         match.Context->getLangOpts());
    };
    auto lessToken = clang::Lexer::findNextToken(
        range.getBegin(), *match.SourceManager, match.Context->getLangOpts());
    while (lessToken && lessToken->getKind() != clang::tok::less) {
      lessToken = nextToken(lessToken);
    }
    if (!lessToken) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "missing '<' token");
    }
    std::optional<Token> endToken = nextToken(lessToken);
    for (auto greaterToken = nextToken(endToken);
         greaterToken && greaterToken->getKind() != clang::tok::greater;
         greaterToken = nextToken(greaterToken)) {
      endToken = greaterToken;
    }
    if (!endToken) {
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "missing '>' token");
    }
    *result += clang::tooling::getText(
        CharSourceRange::getTokenRange(lessToken->getEndLoc(),
                                       endToken->getLastLoc()),
        *match.Context);
    return llvm::Error::success();
  }
  std::string id;
};

Stencil typeAsWritten(StringRef Id) {
  // Using this instead of `describe` so that we get the exact same spelling.
  return std::make_shared<TypeAsWrittenStencil>(std::string(Id));
}

RewriteRuleWith<std::string> MlirOpBuilderCheckRule() {
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
      cat("Use OpType::create(builder, ...) instead of "
          "builder.create<OpType>(...)"));
}
} // namespace

MlirOpBuilderCheck::MlirOpBuilderCheck(StringRef Name, ClangTidyContext *Context)
    : TransformerClangTidyCheck(MlirOpBuilderCheckRule(), Name, Context) {}

} // namespace clang::tidy::mlir_check
