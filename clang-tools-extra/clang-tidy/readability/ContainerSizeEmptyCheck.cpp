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
#include "../utils/OptionsUtils.h"
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
           anyOf(hasParent(
                     mapAnyOf(varDecl, fieldDecl).with(hasType(booleanType()))),
                 hasParent(cxxConstructorDecl(
                     hasAnyConstructorInitializer(cxxCtorInitializer(
                         withInitializer(expr(equalsBoundNode(ExprName))),
                         forField(hasType(booleanType())))))),
                 hasParent(stmt(anyOf(
                     explicitCastExpr(hasDestinationType(booleanType())),
                     mapAnyOf(ifStmt, doStmt, whileStmt, forStmt,
                              conditionalOperator)
                         .with(hasCondition(expr(equalsBoundNode(ExprName)))),
                     parenListExpr(hasParent(varDecl(hasType(booleanType())))),
                     parenExpr(hasParent(
                         explicitCastExpr(hasDestinationType(booleanType())))),
                     returnStmt(forFunction(returns(booleanType()))),
                     cxxUnresolvedConstructExpr(hasType(booleanType())),
                     invocation(hasAnyArgumentWithParam(
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

AST_MATCHER(CXXConstructExpr, isDefaultConstruction) {
  return Node.getConstructor()->isDefaultConstructor();
}

AST_MATCHER(QualType, isIntegralType) {
  return Node->isIntegralType(Finder->getASTContext());
}

AST_MATCHER_P(UserDefinedLiteral, hasLiteral,
              clang::ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  if (const Expr *CookedLiteral = Node.getCookedLiteral()) {
    return InnerMatcher.matches(*CookedLiteral, Finder, Builder);
  }
  return false;
}

} // namespace ast_matchers
namespace tidy::readability {

using utils::isBinaryOrTernary;

ContainerSizeEmptyCheck::ContainerSizeEmptyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ExcludedComparisonTypes(utils::options::parseStringList(
          Options.get("ExcludedComparisonTypes", "::std::array"))) {}

void ContainerSizeEmptyCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ExcludedComparisonTypes",
                utils::options::serializeStringList(ExcludedComparisonTypes));
}

void ContainerSizeEmptyCheck::registerMatchers(MatchFinder *Finder) {
  const auto ValidContainerRecord = cxxRecordDecl(isSameOrDerivedFrom(
      namedDecl(has(cxxMethodDecl(isConst(), parameterCountIs(0), isPublic(),
                                  hasAnyName("size", "length"),
                                  returns(qualType(isIntegralType(),
                                                   unless(booleanType()))))
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

  const auto WrongUse =
      anyOf(hasParent(binaryOperator(
                          isComparisonOperator(),
                          hasEitherOperand(anyOf(integerLiteral(equals(1)),
                                                 integerLiteral(equals(0)))))
                          .bind("SizeBinaryOp")),
            usedInBooleanContext());

  Finder->addMatcher(
      cxxMemberCallExpr(
          on(expr(anyOf(hasType(ValidContainer),
                        hasType(pointsTo(ValidContainer)),
                        hasType(references(ValidContainer))))
                 .bind("MemberCallObject")),
          callee(
              cxxMethodDecl(hasAnyName("size", "length")).bind("SizeMethod")),
          WrongUse,
          unless(hasAncestor(
              cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
          .bind("SizeCallExpr"),
      this);

  Finder->addMatcher(
      callExpr(has(cxxDependentScopeMemberExpr(
                       hasObjectExpression(
                           expr(anyOf(hasType(ValidContainer),
                                      hasType(pointsTo(ValidContainer)),
                                      hasType(references(ValidContainer))))
                               .bind("MemberCallObject")),
                       anyOf(hasMemberName("size"), hasMemberName("length")))
                       .bind("DependentExpr")),
               WrongUse,
               unless(hasAncestor(
                   cxxMethodDecl(ofClass(equalsBoundNode("container"))))))
          .bind("SizeCallExpr"),
      this);

  // Comparison to empty string or empty constructor.
  const auto WrongComparend =
      anyOf(stringLiteral(hasSize(0)),
            userDefinedLiteral(hasLiteral(stringLiteral(hasSize(0)))),
            cxxConstructExpr(isDefaultConstruction()),
            cxxUnresolvedConstructExpr(argumentCountIs(0)));
  // Match the object being compared.
  const auto STLArg =
      anyOf(unaryOperator(
                hasOperatorName("*"),
                hasUnaryOperand(
                    expr(hasType(pointsTo(ValidContainer))).bind("Pointee"))),
            expr(hasType(ValidContainer)).bind("STLObject"));

  const auto ExcludedComparisonTypesMatcher = qualType(anyOf(
      hasDeclaration(
          cxxRecordDecl(matchers::matchesAnyListedName(ExcludedComparisonTypes))
              .bind("excluded")),
      hasCanonicalType(hasDeclaration(
          cxxRecordDecl(matchers::matchesAnyListedName(ExcludedComparisonTypes))
              .bind("excluded")))));
  const auto SameExcludedComparisonTypesMatcher =
      qualType(anyOf(hasDeclaration(cxxRecordDecl(equalsBoundNode("excluded"))),
                     hasCanonicalType(hasDeclaration(
                         cxxRecordDecl(equalsBoundNode("excluded"))))));

  Finder->addMatcher(
      binaryOperation(
          hasAnyOperatorName("==", "!="), hasOperands(WrongComparend, STLArg),
          unless(allOf(hasLHS(hasType(ExcludedComparisonTypesMatcher)),
                       hasRHS(hasType(SameExcludedComparisonTypesMatcher)))),
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
  const auto *BinCmpRewritten =
      Result.Nodes.getNodeAs<CXXRewrittenBinaryOperator>("BinCmp");
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
  const auto *OpCallExpr = dyn_cast<CXXOperatorCallExpr>(E);
  if (isBinaryOrTernary(E) || isa<UnaryOperator>(E) ||
      (OpCallExpr && (OpCallExpr->getOperator() == OO_Star))) {
    ReplacementText = "(" + ReplacementText + ")";
  }
  if (OpCallExpr &&
      OpCallExpr->getOperator() == OverloadedOperatorKind::OO_Arrow) {
    // This can happen if the object is a smart pointer. Don't add anything
    // because a '->' is already there (PR#51776), just call the method.
    ReplacementText += "empty()";
  } else if (E->getType()->isPointerType())
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
  } else if (BinCmpRewritten) {
    if (BinCmpRewritten->getOpcode() == BinaryOperatorKind::BO_NE) {
      ReplacementText = "!" + ReplacementText;
    }
    Hint = FixItHint::CreateReplacement(BinCmpRewritten->getSourceRange(),
                                        ReplacementText);
  } else if (BinaryOp) { // Determine the correct transformation.
    const auto *LiteralLHS =
        llvm::dyn_cast<IntegerLiteral>(BinaryOp->getLHS()->IgnoreImpCasts());
    const auto *LiteralRHS =
        llvm::dyn_cast<IntegerLiteral>(BinaryOp->getRHS()->IgnoreImpCasts());
    const bool ContainerIsLHS = !LiteralLHS;

    uint64_t Value = 0;
    if (LiteralLHS)
      Value = LiteralLHS->getValue().getLimitedValue();
    else if (LiteralRHS)
      Value = LiteralRHS->getValue().getLimitedValue();
    else
      return;

    bool Negation = false;
    const auto OpCode = BinaryOp->getOpcode();

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
    auto Diag = diag(WarnLoc, "the 'empty' method should be used to check "
                              "for emptiness instead of %0");
    if (const auto *SizeMethod =
            Result.Nodes.getNodeAs<NamedDecl>("SizeMethod"))
      Diag << SizeMethod;
    else if (const auto *DependentExpr =
                 Result.Nodes.getNodeAs<CXXDependentScopeMemberExpr>(
                     "DependentExpr"))
      Diag << DependentExpr->getMember();
    else
      Diag << "unknown method";
    Diag << Hint;
  } else {
    WarnLoc = BinCmpTempl
                  ? BinCmpTempl->getBeginLoc()
                  : (BinCmp ? BinCmp->getBeginLoc()
                            : (BinCmpRewritten ? BinCmpRewritten->getBeginLoc()
                                               : SourceLocation{}));
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

} // namespace tidy::readability
} // namespace clang
