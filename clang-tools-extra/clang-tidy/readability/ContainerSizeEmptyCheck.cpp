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
namespace ast_matchers {
AST_POLYMORPHIC_MATCHER_P2(hasAnyArgumentWithParam,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(CallExpr,
                                                           CXXConstructExpr),
                           internal::Matcher<Expr>, ArgMatcher,
                           internal::Matcher<ParmVarDecl>, ParamMatcher) {
  BoundNodesTreeBuilder Result;
  // The first argument of an overloaded member operator is the implicit object
  // argument of the method which should not be matched against a parameter, so
  // we skip over it here.
  BoundNodesTreeBuilder Matches;
  unsigned ArgIndex = cxxOperatorCallExpr(callee(cxxMethodDecl()))
                              .matches(Node, Finder, &Matches)
                          ? 1
                          : 0;
  int ParamIndex = 0;
  for (; ArgIndex < Node.getNumArgs(); ++ArgIndex) {
    BoundNodesTreeBuilder ArgMatches(*Builder);
    if (ArgMatcher.matches(*(Node.getArg(ArgIndex)->IgnoreParenCasts()), Finder,
                           &ArgMatches)) {
      BoundNodesTreeBuilder ParamMatches(ArgMatches);
      if (expr(anyOf(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                         hasParameter(ParamIndex, ParamMatcher)))),
                     callExpr(callee(functionDecl(
                         hasParameter(ParamIndex, ParamMatcher))))))
              .matches(Node, Finder, &ParamMatches)) {
        Result.addMatch(ParamMatches);
        *Builder = std::move(Result);
        return true;
      }
    }
    ++ParamIndex;
  }
  return false;
}

AST_MATCHER(Expr, usedInBooleanContext) {
  const char *ExprName = "__booleanContextExpr";
  auto Result =
      expr(expr().bind(ExprName),
           anyOf(hasParent(varDecl(hasType(booleanType()))),
                 hasParent(cxxConstructorDecl(
                     hasAnyConstructorInitializer(cxxCtorInitializer(
                         withInitializer(expr(equalsBoundNode(ExprName))),
                         forField(hasType(booleanType())))))),
                 hasParent(fieldDecl(hasType(booleanType()))),
                 hasParent(stmt(anyOf(
                     explicitCastExpr(hasDestinationType(booleanType())),
                     ifStmt(hasCondition(expr(equalsBoundNode(ExprName)))),
                     doStmt(hasCondition(expr(equalsBoundNode(ExprName)))),
                     whileStmt(hasCondition(expr(equalsBoundNode(ExprName)))),
                     forStmt(hasCondition(expr(equalsBoundNode(ExprName)))),
                     conditionalOperator(
                         hasCondition(expr(equalsBoundNode(ExprName)))),
                     parenListExpr(hasParent(varDecl(hasType(booleanType())))),
                     parenExpr(hasParent(
                         explicitCastExpr(hasDestinationType(booleanType())))),
                     returnStmt(forFunction(returns(booleanType()))),
                     cxxUnresolvedConstructExpr(hasType(booleanType())),
                     callExpr(hasAnyArgumentWithParam(
                         expr(equalsBoundNode(ExprName)),
                         parmVarDecl(hasType(booleanType())))),
                     binaryOperator(hasAnyOperatorName("&&", "||")),
                     unaryOperator(hasOperatorName("!")).bind("NegOnSize"))))))
          .matches(Node, Finder, Builder);
  Builder->removeBindings([ExprName](const BoundNodesMap &Nodes) {
    return Nodes.getNode(ExprName).getNodeKind().isNone();
  });
  return Result;
}
} // namespace ast_matchers
namespace tidy {
namespace readability {

using utils::IsBinaryOrTernary;

ContainerSizeEmptyCheck::ContainerSizeEmptyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ContainerSizeEmptyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ValidContainerRecord = cxxRecordDecl(isSameOrDerivedFrom(
      namedDecl(
          has(cxxMethodDecl(isConst(), parameterCountIs(0), isPublic(),
                            hasName("size"),
                            returns(qualType(isInteger(), unless(booleanType()),
                                             unless(elaboratedType()))))
                  .bind("size")),
          has(cxxMethodDecl(isConst(), parameterCountIs(0), isPublic(),
                            hasName("empty"), returns(booleanType()))
                  .bind("empty")))
          .bind("container")));

  const auto ValidContainerNonTemplateType =
      qualType(hasUnqualifiedDesugaredType(
          recordType(hasDeclaration(ValidContainerRecord))));
  const auto ValidContainerTemplateType =
      qualType(hasUnqualifiedDesugaredType(templateSpecializationType(
          hasDeclaration(classTemplateDecl(has(ValidContainerRecord))))));

  const auto ValidContainer = qualType(
      anyOf(ValidContainerNonTemplateType, ValidContainerTemplateType));

  const auto WrongUse = traverse(
      TK_AsIs,
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
          usedInBooleanContext()));

  Finder->addMatcher(
      cxxMemberCallExpr(unless(isInTemplateInstantiation()),
                        on(expr(anyOf(hasType(ValidContainer),
                                      hasType(pointsTo(ValidContainer)),
                                      hasType(references(ValidContainer))))
                               .bind("MemberCallObject")),
                        callee(cxxMethodDecl(hasName("size"))), WrongUse,
                        unless(hasAncestor(cxxMethodDecl(
                            ofClass(equalsBoundNode("container"))))))
          .bind("SizeCallExpr"),
      this);

  Finder->addMatcher(
      callExpr(has(cxxDependentScopeMemberExpr(
                   hasObjectExpression(
                       expr(anyOf(hasType(ValidContainer),
                                  hasType(pointsTo(ValidContainer)),
                                  hasType(references(ValidContainer))))
                           .bind("MemberCallObject")),
                   hasMemberName("size"))),
               WrongUse,
               unless(hasAncestor(
                   cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
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
      cxxConstructExpr(hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
                       has(expr(ignoringImpCasts(DefaultConstructor)))),
      cxxConstructExpr(hasDeclaration(cxxConstructorDecl(isMoveConstructor())),
                       has(expr(ignoringImpCasts(DefaultConstructor)))),
      cxxUnresolvedConstructExpr(argumentCountIs(0)));
  // Match the object being compared.
  const auto STLArg =
      anyOf(unaryOperator(
                hasOperatorName("*"),
                hasUnaryOperand(
                    expr(hasType(pointsTo(ValidContainer))).bind("Pointee"))),
            expr(hasType(ValidContainer)).bind("STLObject"));
  Finder->addMatcher(
      cxxOperatorCallExpr(
          unless(isInTemplateInstantiation()),
          hasAnyOverloadedOperatorName("==", "!="),
          anyOf(allOf(hasArgument(0, WrongComparend), hasArgument(1, STLArg)),
                allOf(hasArgument(0, STLArg), hasArgument(1, WrongComparend))),
          unless(hasAncestor(
              cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
          .bind("BinCmp"),
      this);
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("==", "!="),
                     anyOf(allOf(hasLHS(WrongComparend), hasRHS(STLArg)),
                           allOf(hasLHS(STLArg), hasRHS(WrongComparend))),
                     unless(hasAncestor(
                         cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
          .bind("BinCmp"),
      this);
}

void ContainerSizeEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall = Result.Nodes.getNodeAs<Expr>("SizeCallExpr");
  const auto *MemberCallObject =
      Result.Nodes.getNodeAs<Expr>("MemberCallObject");
  const auto *BinCmp = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("BinCmp");
  const auto *BinCmpTempl = Result.Nodes.getNodeAs<BinaryOperator>("BinCmp");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("SizeBinaryOp");
  const auto *Pointee = Result.Nodes.getNodeAs<Expr>("Pointee");
  const auto *E =
      MemberCallObject
          ? MemberCallObject
          : (Pointee ? Pointee : Result.Nodes.getNodeAs<Expr>("STLObject"));
  FixItHint Hint;
  std::string ReplacementText = std::string(
      Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()),
                           *Result.SourceManager, getLangOpts()));
  if (IsBinaryOrTernary(E) || isa<UnaryOperator>(E)) {
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
  } else if (BinCmpTempl) {
    if (BinCmpTempl->getOpcode() == BinaryOperatorKind::BO_NE) {
      ReplacementText = "!" + ReplacementText;
    }
    Hint = FixItHint::CreateReplacement(BinCmpTempl->getSourceRange(),
                                        ReplacementText);
  } else if (BinaryOp) { // Determine the correct transformation.
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

  auto WarnLoc = MemberCall ? MemberCall->getBeginLoc() : SourceLocation{};

  if (WarnLoc.isValid()) {
    diag(WarnLoc, "the 'empty' method should be used to check "
                  "for emptiness instead of 'size'")
        << Hint;
  } else {
    WarnLoc = BinCmpTempl ? BinCmpTempl->getBeginLoc()
                          : (BinCmp ? BinCmp->getBeginLoc() : SourceLocation{});
    diag(WarnLoc, "the 'empty' method should be used to check "
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
