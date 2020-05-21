//===--- ContainerSizeEmptyCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ContainerSizeEmptyCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

using utils::IsBinaryOrTernary;

ContainerSizeEmptyCheck::ContainerSizeEmptyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ContainerSizeEmptyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ValidContainer = qualType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(isSameOrDerivedFrom(
          namedDecl(
              has(cxxMethodDecl(
                      isConst(), parameterCountIs(0), isPublic(),
                      hasName("size"),
                      returns(qualType(isInteger(), unless(booleanType()))))
                      .bind("size")),
              has(cxxMethodDecl(isConst(), parameterCountIs(0), isPublic(),
                                hasName("empty"), returns(booleanType()))
                      .bind("empty")))
              .bind("container")))))));

  const auto WrongUse = traverse(
      ast_type_traits::TK_AsIs,
      anyOf(
          hasParent(binaryOperator(isComparisonOperator(),
                                   hasEitherOperand(ignoringImpCasts(
                                       anyOf(integerLiteral(equals(1)),
                                             integerLiteral(equals(0))))))
                        .bind("SizeBinaryOp")),
          hasParent(implicitCastExpr(
              hasImplicitDestinationType(booleanType()),
              anyOf(hasParent(
                        unaryOperator(hasOperatorName("!")).bind("NegOnSize")),
                    anything()))),
          hasParent(explicitCastExpr(hasDestinationType(booleanType())))));

  Finder->addMatcher(
      cxxMemberCallExpr(on(expr(anyOf(hasType(ValidContainer),
                                      hasType(pointsTo(ValidContainer)),
                                      hasType(references(ValidContainer))))),
                        callee(cxxMethodDecl(hasName("size"))), WrongUse,
                        unless(hasAncestor(cxxMethodDecl(
                            ofClass(equalsBoundNode("container"))))))
          .bind("SizeCallExpr"),
      this);

  // Empty constructor matcher.
  const auto DefaultConstructor = cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(isDefaultConstructor())));
  // Comparison to empty string or empty constructor.
  const auto WrongComparend = anyOf(
      ignoringImpCasts(stringLiteral(hasSize(0))),
      ignoringImpCasts(cxxBindTemporaryExpr(has(DefaultConstructor))),
      ignoringImplicit(DefaultConstructor),
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
          has(expr(ignoringImpCasts(DefaultConstructor)))),
      cxxConstructExpr(
          hasDeclaration(cxxConstructorDecl(isMoveConstructor())),
          has(expr(ignoringImpCasts(DefaultConstructor)))));
  // Match the object being compared.
  const auto STLArg =
      anyOf(unaryOperator(
                hasOperatorName("*"),
                hasUnaryOperand(
                    expr(hasType(pointsTo(ValidContainer))).bind("Pointee"))),
            expr(hasType(ValidContainer)).bind("STLObject"));
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasAnyOverloadedOperatorName("==", "!="),
          anyOf(allOf(hasArgument(0, WrongComparend), hasArgument(1, STLArg)),
                allOf(hasArgument(0, STLArg), hasArgument(1, WrongComparend))),
          unless(hasAncestor(
              cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
          .bind("BinCmp"),
      this);
}

void ContainerSizeEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("SizeCallExpr");
  const auto *BinCmp = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("BinCmp");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("SizeBinaryOp");
  const auto *Pointee = Result.Nodes.getNodeAs<Expr>("Pointee");
  const auto *E =
      MemberCall
          ? MemberCall->getImplicitObjectArgument()
          : (Pointee ? Pointee : Result.Nodes.getNodeAs<Expr>("STLObject"));
  FixItHint Hint;
  std::string ReplacementText = std::string(
      Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()),
                           *Result.SourceManager, getLangOpts()));
  if (BinCmp && IsBinaryOrTernary(E)) {
    // Not just a DeclRefExpr, so parenthesize to be on the safe side.
    ReplacementText = "(" + ReplacementText + ")";
  }
  if (E->getType()->isPointerType())
    ReplacementText += "->empty()";
  else
    ReplacementText += ".empty()";

  if (BinCmp) {
    if (BinCmp->getOperator() == OO_ExclaimEqual) {
      ReplacementText = "!" + ReplacementText;
    }
    Hint =
        FixItHint::CreateReplacement(BinCmp->getSourceRange(), ReplacementText);
  } else if (BinaryOp) {  // Determine the correct transformation.
    bool Negation = false;
    const bool ContainerIsLHS =
        !llvm::isa<IntegerLiteral>(BinaryOp->getLHS()->IgnoreImpCasts());
    const auto OpCode = BinaryOp->getOpcode();
    uint64_t Value = 0;
    if (ContainerIsLHS) {
      if (const auto *Literal = llvm::dyn_cast<IntegerLiteral>(
              BinaryOp->getRHS()->IgnoreImpCasts()))
        Value = Literal->getValue().getLimitedValue();
      else
        return;
    } else {
      Value =
          llvm::dyn_cast<IntegerLiteral>(BinaryOp->getLHS()->IgnoreImpCasts())
              ->getValue()
              .getLimitedValue();
    }

    // Constant that is not handled.
    if (Value > 1)
      return;

    if (Value == 1 && (OpCode == BinaryOperatorKind::BO_EQ ||
                       OpCode == BinaryOperatorKind::BO_NE))
      return;

    // Always true, no warnings for that.
    if ((OpCode == BinaryOperatorKind::BO_GE && Value == 0 && ContainerIsLHS) ||
        (OpCode == BinaryOperatorKind::BO_LE && Value == 0 && !ContainerIsLHS))
      return;

    // Do not warn for size > 1, 1 < size, size <= 1, 1 >= size.
    if (Value == 1) {
      if ((OpCode == BinaryOperatorKind::BO_GT && ContainerIsLHS) ||
          (OpCode == BinaryOperatorKind::BO_LT && !ContainerIsLHS))
        return;
      if ((OpCode == BinaryOperatorKind::BO_LE && ContainerIsLHS) ||
          (OpCode == BinaryOperatorKind::BO_GE && !ContainerIsLHS))
        return;
    }

    if (OpCode == BinaryOperatorKind::BO_NE && Value == 0)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_GT ||
         OpCode == BinaryOperatorKind::BO_GE) &&
        ContainerIsLHS)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_LT ||
         OpCode == BinaryOperatorKind::BO_LE) &&
        !ContainerIsLHS)
      Negation = true;

    if (Negation)
      ReplacementText = "!" + ReplacementText;
    Hint = FixItHint::CreateReplacement(BinaryOp->getSourceRange(),
                                        ReplacementText);

  } else {
    // If there is a conversion above the size call to bool, it is safe to just
    // replace size with empty.
    if (const auto *UnaryOp =
            Result.Nodes.getNodeAs<UnaryOperator>("NegOnSize"))
      Hint = FixItHint::CreateReplacement(UnaryOp->getSourceRange(),
                                          ReplacementText);
    else
      Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                          "!" + ReplacementText);
  }

  if (MemberCall) {
    diag(MemberCall->getBeginLoc(),
         "the 'empty' method should be used to check "
         "for emptiness instead of 'size'")
        << Hint;
  } else {
    diag(BinCmp->getBeginLoc(),
         "the 'empty' method should be used to check "
         "for emptiness instead of comparing to an empty object")
        << Hint;
  }

  const auto *Container = Result.Nodes.getNodeAs<NamedDecl>("container");
  if (const auto *CTS = dyn_cast<ClassTemplateSpecializationDecl>(Container)) {
    // The definition of the empty() method is the same for all implicit
    // instantiations. In order to avoid duplicate or inconsistent warnings
    // (depending on how deduplication is done), we use the same class name
    // for all implicit instantiations of a template.
    if (CTS->getSpecializationKind() == TSK_ImplicitInstantiation)
      Container = CTS->getSpecializedTemplate();
  }
  const auto *Empty = Result.Nodes.getNodeAs<FunctionDecl>("empty");

  diag(Empty->getLocation(), "method %0::empty() defined here",
       DiagnosticIDs::Note)
      << Container;
}

} // namespace readability
} // namespace tidy
} // namespace clang
