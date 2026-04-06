//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerDataPointerCheck.h"

#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

constexpr StringRef ContainerExprName = "container-expr";
constexpr StringRef DerefContainerExprName = "deref-container-expr";
constexpr StringRef AddrOfContainerExprName = "addr-of-container-expr";
constexpr StringRef AddressOfName = "address-of";

void ContainerDataPointerCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredContainers",
                utils::options::serializeStringList(IgnoredContainers));
}

ContainerDataPointerCheck::ContainerDataPointerCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredContainers(utils::options::parseStringList(
          Options.get("IgnoredContainers", ""))) {}

void ContainerDataPointerCheck::registerMatchers(MatchFinder *Finder) {
  const auto Record =
      cxxRecordDecl(
          unless(matchers::matchesAnyListedRegexName(IgnoredContainers)),
          isSameOrDerivedFrom(
              namedDecl(anyOf(has(cxxMethodDecl(isPublic(), hasName("c_str"))
                                      .bind("c_str")),
                              has(cxxMethodDecl(isPublic(), hasName("data")))))
                  .bind("container")))
          .bind("record");

  const auto NonTemplateContainerType =
      qualType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(Record))));
  const auto TemplateContainerType =
      qualType(hasUnqualifiedDesugaredType(templateSpecializationType(
          hasDeclaration(classTemplateDecl(has(Record))))));

  const auto Container =
      qualType(anyOf(NonTemplateContainerType, TemplateContainerType));

  const auto ContainerExpr = anyOf(
      unaryOperator(
          hasOperatorName("*"),
          hasUnaryOperand(
              expr(hasType(pointsTo(Container))).bind(DerefContainerExprName)))
          .bind(ContainerExprName),
      unaryOperator(hasOperatorName("&"),
                    hasUnaryOperand(expr(anyOf(hasType(Container),
                                               hasType(references(Container))))
                                        .bind(AddrOfContainerExprName)))
          .bind(ContainerExprName),
      expr(anyOf(hasType(Container), hasType(pointsTo(Container)),
                 hasType(references(Container))))
          .bind(ContainerExprName));

  const auto Zero = integerLiteral(equals(0));

  const auto SubscriptOperator = callee(cxxMethodDecl(hasName("operator[]")));

  Finder->addMatcher(
      unaryOperator(
          hasOperatorName("&"),
          hasUnaryOperand(expr(
              anyOf(cxxOperatorCallExpr(SubscriptOperator, argumentCountIs(2),
                                        hasArgument(0, ContainerExpr),
                                        hasArgument(1, Zero)),
                    cxxMemberCallExpr(SubscriptOperator, on(ContainerExpr),
                                      argumentCountIs(1), hasArgument(0, Zero)),
                    arraySubscriptExpr(hasLHS(ContainerExpr), hasRHS(Zero))))))
          .bind(AddressOfName),
      this);
}

void ContainerDataPointerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UO = Result.Nodes.getNodeAs<UnaryOperator>(AddressOfName);
  const auto *CE = Result.Nodes.getNodeAs<Expr>(ContainerExprName);
  const auto *DCE = Result.Nodes.getNodeAs<Expr>(DerefContainerExprName);
  const auto *ACE = Result.Nodes.getNodeAs<Expr>(AddrOfContainerExprName);

  const auto *CStrDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("c_str");

  if (!UO || !CE)
    return;

  if (DCE && !CE->getType()->isPointerType())
    CE = DCE;
  else if (ACE)
    CE = ACE;

  SourceRange ReplacementRange = UO->getSourceRange();
  bool UseCStr = false;
  if (CStrDecl) {
    auto Parents = Result.Context->getParents(*UO);

    if (!Parents.empty()) {
      if (const auto *VD = Parents[0].get<VarDecl>()) {
        QualType VarType = VD->getType();
        if (VarType->isPointerType()) {
          QualType PointeeType = VarType->getPointeeType();
          UseCStr = PointeeType.isConstQualified();
        }
      } else if (const auto *ICE = Parents[0].get<ImplicitCastExpr>()) {
        QualType CastType = ICE->getType();
        if (CastType->isPointerType()) {
          QualType PointeeType = CastType->getPointeeType();
          UseCStr = PointeeType.isConstQualified();
        }
      } else if (const auto *Cast = Parents[0].get<CastExpr>()) {
        QualType CastType = Cast->getType();
        if (CastType->isPointerType()) {
          QualType PointeeType = CastType->getPointeeType();
          UseCStr = PointeeType.isConstQualified();
          if (UseCStr) {
            // if it's a const cast, use the Cast range as replacement range
            // e.g. (const char*)&s[0] -> s.c_str()
            ReplacementRange = Cast->getSourceRange();
          }
        }
      }
    }

    if (!UseCStr) {
      QualType ContainerType = CE->getType();
      if (ContainerType->isPointerType())
        ContainerType = ContainerType->getPointeeType();
      UseCStr = ContainerType.isConstQualified();
    }
  }

  const SourceRange SrcRange = CE->getSourceRange();

  std::string ReplacementText{
      Lexer::getSourceText(CharSourceRange::getTokenRange(SrcRange),
                           *Result.SourceManager, getLangOpts())};

  const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(CE);
  const bool NeedsParens =
      OpCall ? (OpCall->getOperator() != OO_Subscript)
             : !isa<DeclRefExpr, MemberExpr, ArraySubscriptExpr, CallExpr>(CE);
  if (NeedsParens)
    ReplacementText = "(" + ReplacementText + ")";

  ReplacementText += CE->getType()->isPointerType() ? "->" : ".";
  ReplacementText += UseCStr ? "c_str()" : "data()";

  const FixItHint Hint =
      FixItHint::CreateReplacement(ReplacementRange, ReplacementText);
  diag(UO->getBeginLoc(),
       "'%select{data|c_str}0' should be used for accessing the data pointer "
       "instead of taking the address of the 0-th element")
      << UseCStr << Hint;
}
} // namespace clang::tidy::readability
