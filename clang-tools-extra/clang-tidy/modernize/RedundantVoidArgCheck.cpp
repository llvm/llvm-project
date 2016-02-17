//===- RedundantVoidArgCheck.cpp - clang-tidy -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantVoidArgCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {

namespace {

// Determine if the given QualType is a nullary function or pointer to same.
bool protoTypeHasNoParms(QualType QT) {
  if (auto PT = QT->getAs<PointerType>()) {
    QT = PT->getPointeeType();
  }
  if (auto *MPT = QT->getAs<MemberPointerType>()) {
    QT = MPT->getPointeeType();
  }
  if (auto FP = QT->getAs<FunctionProtoType>()) {
    return FP->getNumParams() == 0;
  }
  return false;
}

const char FunctionId[] = "function";
const char TypedefId[] = "typedef";
const char FieldId[] = "field";
const char VarId[] = "var";
const char NamedCastId[] = "named-cast";
const char CStyleCastId[] = "c-style-cast";
const char ExplicitCastId[] = "explicit-cast";
const char LambdaId[] = "lambda";

} // namespace

namespace tidy {
namespace modernize {

void RedundantVoidArgCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(parameterCountIs(0), unless(isImplicit()),
                                  unless(isExternC()))
                         .bind(FunctionId),
                     this);
  Finder->addMatcher(typedefDecl().bind(TypedefId), this);
  auto ParenFunctionType = parenType(innerType(functionType()));
  auto PointerToFunctionType = pointee(ParenFunctionType);
  auto FunctionOrMemberPointer =
      anyOf(hasType(pointerType(PointerToFunctionType)),
            hasType(memberPointerType(PointerToFunctionType)));
  Finder->addMatcher(fieldDecl(FunctionOrMemberPointer).bind(FieldId), this);
  Finder->addMatcher(varDecl(FunctionOrMemberPointer).bind(VarId), this);
  auto CastDestinationIsFunction =
      hasDestinationType(pointsTo(ParenFunctionType));
  Finder->addMatcher(
      cStyleCastExpr(CastDestinationIsFunction).bind(CStyleCastId), this);
  Finder->addMatcher(
      cxxStaticCastExpr(CastDestinationIsFunction).bind(NamedCastId), this);
  Finder->addMatcher(
      cxxReinterpretCastExpr(CastDestinationIsFunction).bind(NamedCastId),
      this);
  Finder->addMatcher(
      cxxConstCastExpr(CastDestinationIsFunction).bind(NamedCastId), this);
  Finder->addMatcher(lambdaExpr().bind(LambdaId), this);
}

void RedundantVoidArgCheck::check(const MatchFinder::MatchResult &Result) {
  if (!Result.Context->getLangOpts().CPlusPlus) {
    return;
  }

  const BoundNodes &Nodes = Result.Nodes;
  if (const auto *Function = Nodes.getNodeAs<FunctionDecl>(FunctionId)) {
    processFunctionDecl(Result, Function);
  } else if (const auto *Typedef = Nodes.getNodeAs<TypedefDecl>(TypedefId)) {
    processTypedefDecl(Result, Typedef);
  } else if (const auto *Member = Nodes.getNodeAs<FieldDecl>(FieldId)) {
    processFieldDecl(Result, Member);
  } else if (const auto *Var = Nodes.getNodeAs<VarDecl>(VarId)) {
    processVarDecl(Result, Var);
  } else if (const auto *NamedCast =
                 Nodes.getNodeAs<CXXNamedCastExpr>(NamedCastId)) {
    processNamedCastExpr(Result, NamedCast);
  } else if (const auto *CStyleCast =
                 Nodes.getNodeAs<CStyleCastExpr>(CStyleCastId)) {
    processExplicitCastExpr(Result, CStyleCast);
  } else if (const auto *ExplicitCast =
                 Nodes.getNodeAs<ExplicitCastExpr>(ExplicitCastId)) {
    processExplicitCastExpr(Result, ExplicitCast);
  } else if (const auto *Lambda = Nodes.getNodeAs<LambdaExpr>(LambdaId)) {
    processLambdaExpr(Result, Lambda);
  }
}

void RedundantVoidArgCheck::processFunctionDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *Function) {
  SourceLocation Start = Function->getLocStart();
  if (Function->isThisDeclarationADefinition()) {
    SourceLocation End;
    if (Function->hasBody())
      End = Function->getBody()->getLocStart().getLocWithOffset(-1);
    else
      End = Function->getLocEnd();
    removeVoidArgumentTokens(Result, SourceRange(Start, End),
                             "function definition");
  } else {
    removeVoidArgumentTokens(Result, Function->getSourceRange(),
                             "function declaration");
  }
}

void RedundantVoidArgCheck::removeVoidArgumentTokens(
    const ast_matchers::MatchFinder::MatchResult &Result, SourceRange Range,
    StringRef GrammarLocation) {
  CharSourceRange CharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Range), *Result.SourceManager,
      Result.Context->getLangOpts());

  std::string DeclText = Lexer::getSourceText(CharRange, *Result.SourceManager,
                                              Result.Context->getLangOpts())
                             .str();
  Lexer PrototypeLexer(CharRange.getBegin(), Result.Context->getLangOpts(),
                       DeclText.data(), DeclText.data(),
                       DeclText.data() + DeclText.size());
  enum TokenState {
    NothingYet,
    SawLeftParen,
    SawVoid,
  };
  TokenState State = NothingYet;
  Token VoidToken;
  Token ProtoToken;
  std::string Diagnostic =
      ("redundant void argument list in " + GrammarLocation).str();

  while (!PrototypeLexer.LexFromRawLexer(ProtoToken)) {
    switch (State) {
    case NothingYet:
      if (ProtoToken.is(tok::TokenKind::l_paren)) {
        State = SawLeftParen;
      }
      break;
    case SawLeftParen:
      if (ProtoToken.is(tok::TokenKind::raw_identifier) &&
          ProtoToken.getRawIdentifier() == "void") {
        State = SawVoid;
        VoidToken = ProtoToken;
      } else {
        State = NothingYet;
      }
      break;
    case SawVoid:
      State = NothingYet;
      if (ProtoToken.is(tok::TokenKind::r_paren)) {
        removeVoidToken(VoidToken, Diagnostic);
      } else if (ProtoToken.is(tok::TokenKind::l_paren)) {
        State = SawLeftParen;
      }
      break;
    }
  }

  if (State == SawVoid && ProtoToken.is(tok::TokenKind::r_paren)) {
    removeVoidToken(VoidToken, Diagnostic);
  }
}

void RedundantVoidArgCheck::removeVoidToken(Token VoidToken,
                                            StringRef Diagnostic) {
  SourceLocation VoidLoc(VoidToken.getLocation());
  auto VoidRange =
      CharSourceRange::getTokenRange(VoidLoc, VoidLoc.getLocWithOffset(3));
  diag(VoidLoc, Diagnostic) << FixItHint::CreateRemoval(VoidRange);
}

void RedundantVoidArgCheck::processTypedefDecl(
    const MatchFinder::MatchResult &Result, const TypedefDecl *Typedef) {
  if (protoTypeHasNoParms(Typedef->getUnderlyingType())) {
    removeVoidArgumentTokens(Result, Typedef->getSourceRange(), "typedef");
  }
}

void RedundantVoidArgCheck::processFieldDecl(
    const MatchFinder::MatchResult &Result, const FieldDecl *Member) {
  if (protoTypeHasNoParms(Member->getType())) {
    removeVoidArgumentTokens(Result, Member->getSourceRange(),
                             "field declaration");
  }
}

void RedundantVoidArgCheck::processVarDecl(
    const MatchFinder::MatchResult &Result, const VarDecl *Var) {
  if (protoTypeHasNoParms(Var->getType())) {
    SourceLocation Begin = Var->getLocStart();
    if (Var->hasInit()) {
      SourceLocation InitStart =
          Result.SourceManager->getExpansionLoc(Var->getInit()->getLocStart())
              .getLocWithOffset(-1);
      removeVoidArgumentTokens(Result, SourceRange(Begin, InitStart),
                               "variable declaration with initializer");
    } else {
      removeVoidArgumentTokens(Result, Var->getSourceRange(),
                               "variable declaration");
    }
  }
}

void RedundantVoidArgCheck::processNamedCastExpr(
    const MatchFinder::MatchResult &Result, const CXXNamedCastExpr *NamedCast) {
  if (protoTypeHasNoParms(NamedCast->getTypeAsWritten())) {
    removeVoidArgumentTokens(
        Result,
        NamedCast->getTypeInfoAsWritten()->getTypeLoc().getSourceRange(),
        "named cast");
  }
}

void RedundantVoidArgCheck::processExplicitCastExpr(
    const MatchFinder::MatchResult &Result,
    const ExplicitCastExpr *ExplicitCast) {
  if (protoTypeHasNoParms(ExplicitCast->getTypeAsWritten())) {
    removeVoidArgumentTokens(Result, ExplicitCast->getSourceRange(),
                             "cast expression");
  }
}

void RedundantVoidArgCheck::processLambdaExpr(
    const MatchFinder::MatchResult &Result, const LambdaExpr *Lambda) {
  if (Lambda->getLambdaClass()->getLambdaCallOperator()->getNumParams() == 0 &&
      Lambda->hasExplicitParameters()) {
    SourceLocation Begin =
        Lambda->getIntroducerRange().getEnd().getLocWithOffset(1);
    SourceLocation End = Lambda->getBody()->getLocStart().getLocWithOffset(-1);
    removeVoidArgumentTokens(Result, SourceRange(Begin, End),
                             "lambda expression");
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
