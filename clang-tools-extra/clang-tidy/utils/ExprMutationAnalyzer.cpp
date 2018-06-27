//===---------- ExprMutationAnalyzer.cpp - clang-tidy ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ExprMutationAnalyzer.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace tidy {
namespace utils {
using namespace ast_matchers;

namespace {

AST_MATCHER_P(LambdaExpr, hasCaptureInit, const Expr *, E) {
  return llvm::is_contained(Node.capture_inits(), E);
}

AST_MATCHER_P(CXXForRangeStmt, hasRangeStmt,
              ast_matchers::internal::Matcher<DeclStmt>, InnerMatcher) {
  const DeclStmt *const Range = Node.getRangeStmt();
  return InnerMatcher.matches(*Range, Finder, Builder);
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt, CXXTypeidExpr>
    cxxTypeidExpr;

AST_MATCHER(CXXTypeidExpr, isPotentiallyEvaluated) {
  return Node.isPotentiallyEvaluated();
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt, CXXNoexceptExpr>
    cxxNoexceptExpr;

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt,
                                                          GenericSelectionExpr>
    genericSelectionExpr;

AST_MATCHER_P(GenericSelectionExpr, hasControllingExpr,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getControllingExpr(), Finder, Builder);
}

const auto nonConstReferenceType = [] {
  return referenceType(pointee(unless(isConstQualified())));
};

} // namespace

const Stmt *ExprMutationAnalyzer::findMutation(const Expr *Exp) {
  const auto Memoized = Results.find(Exp);
  if (Memoized != Results.end())
    return Memoized->second;

  if (isUnevaluated(Exp))
    return Results[Exp] = nullptr;

  for (const auto &Finder : {&ExprMutationAnalyzer::findDirectMutation,
                             &ExprMutationAnalyzer::findMemberMutation,
                             &ExprMutationAnalyzer::findArrayElementMutation,
                             &ExprMutationAnalyzer::findCastMutation,
                             &ExprMutationAnalyzer::findRangeLoopMutation,
                             &ExprMutationAnalyzer::findReferenceMutation}) {
    if (const Stmt *S = (this->*Finder)(Exp))
      return Results[Exp] = S;
  }

  return Results[Exp] = nullptr;
}

bool ExprMutationAnalyzer::isUnevaluated(const Expr *Exp) {
  return selectFirst<Expr>(
             "expr",
             match(
                 findAll(
                     expr(equalsNode(Exp),
                          anyOf(
                              // `Exp` is part of the underlying expression of
                              // decltype/typeof if it has an ancestor of
                              // typeLoc.
                              hasAncestor(typeLoc(unless(
                                  hasAncestor(unaryExprOrTypeTraitExpr())))),
                              hasAncestor(expr(anyOf(
                                  // `UnaryExprOrTypeTraitExpr` is unevaluated
                                  // unless it's sizeof on VLA.
                                  unaryExprOrTypeTraitExpr(unless(sizeOfExpr(
                                      hasArgumentOfType(variableArrayType())))),
                                  // `CXXTypeidExpr` is unevaluated unless it's
                                  // applied to an expression of glvalue of
                                  // polymorphic class type.
                                  cxxTypeidExpr(
                                      unless(isPotentiallyEvaluated())),
                                  // The controlling expression of
                                  // `GenericSelectionExpr` is unevaluated.
                                  genericSelectionExpr(hasControllingExpr(
                                      hasDescendant(equalsNode(Exp)))),
                                  cxxNoexceptExpr())))))
                         .bind("expr")),
                 *Stm, *Context)) != nullptr;
}

const Stmt *
ExprMutationAnalyzer::findExprMutation(ArrayRef<BoundNodes> Matches) {
  for (const auto &Nodes : Matches) {
    if (const Stmt *S = findMutation(Nodes.getNodeAs<Expr>("expr")))
      return S;
  }
  return nullptr;
}

const Stmt *
ExprMutationAnalyzer::findDeclMutation(ArrayRef<BoundNodes> Matches) {
  for (const auto &DeclNodes : Matches) {
    if (const Stmt *S = findDeclMutation(DeclNodes.getNodeAs<Decl>("decl")))
      return S;
  }
  return nullptr;
}

const Stmt *ExprMutationAnalyzer::findDeclMutation(const Decl *Dec) {
  const auto Refs = match(
      findAll(declRefExpr(to(equalsNode(Dec))).bind("expr")), *Stm, *Context);
  for (const auto &RefNodes : Refs) {
    const auto *E = RefNodes.getNodeAs<Expr>("expr");
    if (findMutation(E))
      return E;
  }
  return nullptr;
}

const Stmt *ExprMutationAnalyzer::findDirectMutation(const Expr *Exp) {
  // LHS of any assignment operators.
  const auto AsAssignmentLhs =
      binaryOperator(isAssignmentOperator(), hasLHS(equalsNode(Exp)));

  // Operand of increment/decrement operators.
  const auto AsIncDecOperand =
      unaryOperator(anyOf(hasOperatorName("++"), hasOperatorName("--")),
                    hasUnaryOperand(equalsNode(Exp)));

  // Invoking non-const member function.
  const auto NonConstMethod = cxxMethodDecl(unless(isConst()));
  const auto AsNonConstThis =
      expr(anyOf(cxxMemberCallExpr(callee(NonConstMethod), on(equalsNode(Exp))),
                 cxxOperatorCallExpr(callee(NonConstMethod),
                                     hasArgument(0, equalsNode(Exp)))));

  // Taking address of 'Exp'.
  // We're assuming 'Exp' is mutated as soon as its address is taken, though in
  // theory we can follow the pointer and see whether it escaped `Stm` or is
  // dereferenced and then mutated. This is left for future improvements.
  const auto AsAmpersandOperand =
      unaryOperator(hasOperatorName("&"),
                    // A NoOp implicit cast is adding const.
                    unless(hasParent(implicitCastExpr(hasCastKind(CK_NoOp)))),
                    hasUnaryOperand(equalsNode(Exp)));
  const auto AsPointerFromArrayDecay =
      castExpr(hasCastKind(CK_ArrayToPointerDecay),
               unless(hasParent(arraySubscriptExpr())), has(equalsNode(Exp)));

  // Used as non-const-ref argument when calling a function.
  const auto NonConstRefParam = forEachArgumentWithParam(
      equalsNode(Exp), parmVarDecl(hasType(nonConstReferenceType())));
  const auto AsNonConstRefArg =
      anyOf(callExpr(NonConstRefParam), cxxConstructExpr(NonConstRefParam));

  // Captured by a lambda by reference.
  // If we're initializing a capture with 'Exp' directly then we're initializing
  // a reference capture.
  // For value captures there will be an ImplicitCastExpr <LValueToRValue>.
  const auto AsLambdaRefCaptureInit = lambdaExpr(hasCaptureInit(Exp));

  // Returned as non-const-ref.
  // If we're returning 'Exp' directly then it's returned as non-const-ref.
  // For returning by value there will be an ImplicitCastExpr <LValueToRValue>.
  // For returning by const-ref there will be an ImplicitCastExpr <NoOp> (for
  // adding const.)
  const auto AsNonConstRefReturn = returnStmt(hasReturnValue(equalsNode(Exp)));

  const auto Matches =
      match(findAll(stmt(anyOf(AsAssignmentLhs, AsIncDecOperand, AsNonConstThis,
                               AsAmpersandOperand, AsPointerFromArrayDecay,
                               AsNonConstRefArg, AsLambdaRefCaptureInit,
                               AsNonConstRefReturn))
                        .bind("stmt")),
            *Stm, *Context);
  return selectFirst<Stmt>("stmt", Matches);
}

const Stmt *ExprMutationAnalyzer::findMemberMutation(const Expr *Exp) {
  // Check whether any member of 'Exp' is mutated.
  const auto MemberExprs = match(
      findAll(memberExpr(hasObjectExpression(equalsNode(Exp))).bind("expr")),
      *Stm, *Context);
  return findExprMutation(MemberExprs);
}

const Stmt *ExprMutationAnalyzer::findArrayElementMutation(const Expr *Exp) {
  // Check whether any element of an array is mutated.
  const auto SubscriptExprs = match(
      findAll(arraySubscriptExpr(hasBase(ignoringImpCasts(equalsNode(Exp))))
                  .bind("expr")),
      *Stm, *Context);
  return findExprMutation(SubscriptExprs);
}

const Stmt *ExprMutationAnalyzer::findCastMutation(const Expr *Exp) {
  // If 'Exp' is casted to any non-const reference type, check the castExpr.
  const auto Casts =
      match(findAll(castExpr(hasSourceExpression(equalsNode(Exp)),
                             anyOf(explicitCastExpr(hasDestinationType(
                                       nonConstReferenceType())),
                                   implicitCastExpr(hasImplicitDestinationType(
                                       nonConstReferenceType()))))
                        .bind("expr")),
            *Stm, *Context);
  return findExprMutation(Casts);
}

const Stmt *ExprMutationAnalyzer::findRangeLoopMutation(const Expr *Exp) {
  // If range for looping over 'Exp' with a non-const reference loop variable,
  // check all declRefExpr of the loop variable.
  const auto LoopVars =
      match(findAll(cxxForRangeStmt(
                hasLoopVariable(
                    varDecl(hasType(nonConstReferenceType())).bind("decl")),
                hasRangeInit(equalsNode(Exp)))),
            *Stm, *Context);
  return findDeclMutation(LoopVars);
}

const Stmt *ExprMutationAnalyzer::findReferenceMutation(const Expr *Exp) {
  // If 'Exp' is bound to a non-const reference, check all declRefExpr to that.
  const auto Refs = match(
      stmt(forEachDescendant(
          varDecl(
              hasType(nonConstReferenceType()),
              hasInitializer(anyOf(equalsNode(Exp),
                                   conditionalOperator(anyOf(
                                       hasTrueExpression(equalsNode(Exp)),
                                       hasFalseExpression(equalsNode(Exp)))))),
              hasParent(declStmt().bind("stmt")),
              // Don't follow the reference in range statement, we've handled
              // that separately.
              unless(hasParent(declStmt(hasParent(
                  cxxForRangeStmt(hasRangeStmt(equalsBoundNode("stmt"))))))))
              .bind("decl"))),
      *Stm, *Context);
  return findDeclMutation(Refs);
}

} // namespace utils
} // namespace tidy
} // namespace clang
