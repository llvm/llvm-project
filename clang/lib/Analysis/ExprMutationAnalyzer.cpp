//===---------- ExprMutationAnalyzer.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
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

AST_MATCHER_P(Expr, maybeEvalCommaExpr,
             ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  const Expr* Result = &Node;
  while (const auto *BOComma =
               dyn_cast_or_null<BinaryOperator>(Result->IgnoreParens())) {
    if (!BOComma->isCommaOp())
      break;
    Result = BOComma->getRHS();
  }
  return InnerMatcher.matches(*Result, Finder, Builder);
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt, CXXTypeidExpr>
    cxxTypeidExpr;

AST_MATCHER(CXXTypeidExpr, isPotentiallyEvaluated) {
  return Node.isPotentiallyEvaluated();
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt,
                                                          GenericSelectionExpr>
    genericSelectionExpr;

AST_MATCHER_P(GenericSelectionExpr, hasControllingExpr,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getControllingExpr(), Finder, Builder);
}

const auto nonConstReferenceType = [] {
  return hasUnqualifiedDesugaredType(
      referenceType(pointee(unless(isConstQualified()))));
};

const auto nonConstPointerType = [] {
  return hasUnqualifiedDesugaredType(
      pointerType(pointee(unless(isConstQualified()))));
};

const auto isMoveOnly = [] {
  return cxxRecordDecl(
      hasMethod(cxxConstructorDecl(isMoveConstructor(), unless(isDeleted()))),
      hasMethod(cxxMethodDecl(isMoveAssignmentOperator(), unless(isDeleted()))),
      unless(anyOf(hasMethod(cxxConstructorDecl(isCopyConstructor(),
                                                unless(isDeleted()))),
                   hasMethod(cxxMethodDecl(isCopyAssignmentOperator(),
                                           unless(isDeleted()))))));
};

template <class T> struct NodeID;
template <> struct NodeID<Expr> { static constexpr StringRef value = "expr"; };
template <> struct NodeID<Decl> { static constexpr StringRef value = "decl"; };
constexpr StringRef NodeID<Expr>::value;
constexpr StringRef NodeID<Decl>::value;

template <class T, class F = const Stmt *(ExprMutationAnalyzer::*)(const T *)>
const Stmt *tryEachMatch(ArrayRef<ast_matchers::BoundNodes> Matches,
                         ExprMutationAnalyzer *Analyzer, F Finder) {
  const StringRef ID = NodeID<T>::value;
  for (const auto &Nodes : Matches) {
    if (const Stmt *S = (Analyzer->*Finder)(Nodes.getNodeAs<T>(ID)))
      return S;
  }
  return nullptr;
}

} // namespace

const Stmt *ExprMutationAnalyzer::findMutation(const Expr *Exp) {
  return findMutationMemoized(Exp,
                              {&ExprMutationAnalyzer::findDirectMutation,
                               &ExprMutationAnalyzer::findMemberMutation,
                               &ExprMutationAnalyzer::findArrayElementMutation,
                               &ExprMutationAnalyzer::findCastMutation,
                               &ExprMutationAnalyzer::findRangeLoopMutation,
                               &ExprMutationAnalyzer::findReferenceMutation,
                               &ExprMutationAnalyzer::findFunctionArgMutation},
                              Results);
}

const Stmt *ExprMutationAnalyzer::findMutation(const Decl *Dec) {
  return tryEachDeclRef(Dec, &ExprMutationAnalyzer::findMutation);
}

const Stmt *ExprMutationAnalyzer::findPointeeMutation(const Expr *Exp) {
  return findMutationMemoized(Exp, {/*TODO*/}, PointeeResults);
}

const Stmt *ExprMutationAnalyzer::findPointeeMutation(const Decl *Dec) {
  return tryEachDeclRef(Dec, &ExprMutationAnalyzer::findPointeeMutation);
}

const Stmt *ExprMutationAnalyzer::findMutationMemoized(
    const Expr *Exp, llvm::ArrayRef<MutationFinder> Finders,
    ResultMap &MemoizedResults) {
  const auto Memoized = MemoizedResults.find(Exp);
  if (Memoized != MemoizedResults.end())
    return Memoized->second;

  if (isUnevaluated(Exp))
    return MemoizedResults[Exp] = nullptr;

  for (const auto &Finder : Finders) {
    if (const Stmt *S = (this->*Finder)(Exp))
      return MemoizedResults[Exp] = S;
  }

  return MemoizedResults[Exp] = nullptr;
}

const Stmt *ExprMutationAnalyzer::tryEachDeclRef(const Decl *Dec,
                                                 MutationFinder Finder) {
  const auto Refs =
      match(findAll(declRefExpr(to(equalsNode(Dec))).bind(NodeID<Expr>::value)),
            Stm, Context);
  for (const auto &RefNodes : Refs) {
    const auto *E = RefNodes.getNodeAs<Expr>(NodeID<Expr>::value);
    if ((this->*Finder)(E))
      return E;
  }
  return nullptr;
}

bool ExprMutationAnalyzer::isUnevaluated(const Expr *Exp) {
  return selectFirst<Expr>(
             NodeID<Expr>::value,
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
                         .bind(NodeID<Expr>::value)),
                 Stm, Context)) != nullptr;
}

const Stmt *
ExprMutationAnalyzer::findExprMutation(ArrayRef<BoundNodes> Matches) {
  return tryEachMatch<Expr>(Matches, this, &ExprMutationAnalyzer::findMutation);
}

const Stmt *
ExprMutationAnalyzer::findDeclMutation(ArrayRef<BoundNodes> Matches) {
  return tryEachMatch<Decl>(Matches, this, &ExprMutationAnalyzer::findMutation);
}

const Stmt *ExprMutationAnalyzer::findExprPointeeMutation(
    ArrayRef<ast_matchers::BoundNodes> Matches) {
  return tryEachMatch<Expr>(Matches, this,
                            &ExprMutationAnalyzer::findPointeeMutation);
}

const Stmt *ExprMutationAnalyzer::findDeclPointeeMutation(
    ArrayRef<ast_matchers::BoundNodes> Matches) {
  return tryEachMatch<Decl>(Matches, this,
                            &ExprMutationAnalyzer::findPointeeMutation);
}

const Stmt *ExprMutationAnalyzer::findDirectMutation(const Expr *Exp) {
  // LHS of any assignment operators.
  const auto AsAssignmentLhs =
      binaryOperator(isAssignmentOperator(),
                     hasLHS(maybeEvalCommaExpr(equalsNode(Exp))));

  // Operand of increment/decrement operators.
  const auto AsIncDecOperand =
      unaryOperator(anyOf(hasOperatorName("++"), hasOperatorName("--")),
                    hasUnaryOperand(maybeEvalCommaExpr(equalsNode(Exp))));

  // Invoking non-const member function.
  // A member function is assumed to be non-const when it is unresolved.
  const auto NonConstMethod = cxxMethodDecl(unless(isConst()));
  const auto AsNonConstThis =
      expr(anyOf(cxxMemberCallExpr(callee(NonConstMethod),
                                   on(maybeEvalCommaExpr(equalsNode(Exp)))),
                 cxxOperatorCallExpr(callee(NonConstMethod),
                                     hasArgument(0,
                                                 maybeEvalCommaExpr(equalsNode(Exp)))),
                 callExpr(callee(expr(anyOf(
                     unresolvedMemberExpr(
                       hasObjectExpression(maybeEvalCommaExpr(equalsNode(Exp)))),
                     cxxDependentScopeMemberExpr(
                         hasObjectExpression(maybeEvalCommaExpr(equalsNode(Exp))))))))));

  // Taking address of 'Exp'.
  // We're assuming 'Exp' is mutated as soon as its address is taken, though in
  // theory we can follow the pointer and see whether it escaped `Stm` or is
  // dereferenced and then mutated. This is left for future improvements.
  const auto AsAmpersandOperand =
      unaryOperator(hasOperatorName("&"),
                    // A NoOp implicit cast is adding const.
                    unless(hasParent(implicitCastExpr(hasCastKind(CK_NoOp)))),
                    hasUnaryOperand(maybeEvalCommaExpr(equalsNode(Exp))));
  const auto AsPointerFromArrayDecay =
      castExpr(hasCastKind(CK_ArrayToPointerDecay),
               unless(hasParent(arraySubscriptExpr())),
               has(maybeEvalCommaExpr(equalsNode(Exp))));
  // Treat calling `operator->()` of move-only classes as taking address.
  // These are typically smart pointers with unique ownership so we treat
  // mutation of pointee as mutation of the smart pointer itself.
  const auto AsOperatorArrowThis =
      cxxOperatorCallExpr(hasOverloadedOperatorName("->"),
                          callee(cxxMethodDecl(ofClass(isMoveOnly()),
                                               returns(nonConstPointerType()))),
                          argumentCountIs(1),
                          hasArgument(0, maybeEvalCommaExpr(equalsNode(Exp))));

  // Used as non-const-ref argument when calling a function.
  // An argument is assumed to be non-const-ref when the function is unresolved.
  // Instantiated template functions are not handled here but in
  // findFunctionArgMutation which has additional smarts for handling forwarding
  // references.
  const auto NonConstRefParam = forEachArgumentWithParam(
      maybeEvalCommaExpr(equalsNode(Exp)),
      parmVarDecl(hasType(nonConstReferenceType())));
  const auto NotInstantiated = unless(hasDeclaration(isInstantiated()));
  const auto AsNonConstRefArg = anyOf(
      callExpr(NonConstRefParam, NotInstantiated),
      cxxConstructExpr(NonConstRefParam, NotInstantiated),
      callExpr(callee(expr(anyOf(unresolvedLookupExpr(), unresolvedMemberExpr(),
                                 cxxDependentScopeMemberExpr(),
                                 hasType(templateTypeParmType())))),
               hasAnyArgument(maybeEvalCommaExpr(equalsNode(Exp)))),
      cxxUnresolvedConstructExpr(hasAnyArgument(maybeEvalCommaExpr(equalsNode(Exp)))));

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
  const auto AsNonConstRefReturn = returnStmt(hasReturnValue(
                                                maybeEvalCommaExpr(equalsNode(Exp))));

  const auto Matches = match(
      traverse(
          ast_type_traits::TK_AsIs,
          findAll(stmt(anyOf(AsAssignmentLhs, AsIncDecOperand, AsNonConstThis,
                             AsAmpersandOperand, AsPointerFromArrayDecay,
                             AsOperatorArrowThis, AsNonConstRefArg,
                             AsLambdaRefCaptureInit, AsNonConstRefReturn))
                      .bind("stmt"))),
      Stm, Context);
  return selectFirst<Stmt>("stmt", Matches);
}

const Stmt *ExprMutationAnalyzer::findMemberMutation(const Expr *Exp) {
  // Check whether any member of 'Exp' is mutated.
  const auto MemberExprs =
      match(findAll(expr(anyOf(memberExpr(hasObjectExpression(equalsNode(Exp))),
                               cxxDependentScopeMemberExpr(
                                   hasObjectExpression(equalsNode(Exp)))))
                        .bind(NodeID<Expr>::value)),
            Stm, Context);
  return findExprMutation(MemberExprs);
}

const Stmt *ExprMutationAnalyzer::findArrayElementMutation(const Expr *Exp) {
  // Check whether any element of an array is mutated.
  const auto SubscriptExprs = match(
      findAll(arraySubscriptExpr(hasBase(ignoringImpCasts(equalsNode(Exp))))
                  .bind(NodeID<Expr>::value)),
      Stm, Context);
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
                        .bind(NodeID<Expr>::value)),
            Stm, Context);
  if (const Stmt *S = findExprMutation(Casts))
    return S;
  // Treat std::{move,forward} as cast.
  const auto Calls =
      match(findAll(callExpr(callee(namedDecl(
                                 hasAnyName("::std::move", "::std::forward"))),
                             hasArgument(0, equalsNode(Exp)))
                        .bind("expr")),
            Stm, Context);
  return findExprMutation(Calls);
}

const Stmt *ExprMutationAnalyzer::findRangeLoopMutation(const Expr *Exp) {
  // If range for looping over 'Exp' with a non-const reference loop variable,
  // check all declRefExpr of the loop variable.
  const auto LoopVars =
      match(findAll(cxxForRangeStmt(
                hasLoopVariable(varDecl(hasType(nonConstReferenceType()))
                                    .bind(NodeID<Decl>::value)),
                hasRangeInit(equalsNode(Exp)))),
            Stm, Context);
  return findDeclMutation(LoopVars);
}

const Stmt *ExprMutationAnalyzer::findReferenceMutation(const Expr *Exp) {
  // Follow non-const reference returned by `operator*()` of move-only classes.
  // These are typically smart pointers with unique ownership so we treat
  // mutation of pointee as mutation of the smart pointer itself.
  const auto Ref =
      match(findAll(cxxOperatorCallExpr(
                        hasOverloadedOperatorName("*"),
                        callee(cxxMethodDecl(ofClass(isMoveOnly()),
                                             returns(nonConstReferenceType()))),
                        argumentCountIs(1), hasArgument(0, equalsNode(Exp)))
                        .bind(NodeID<Expr>::value)),
            Stm, Context);
  if (const Stmt *S = findExprMutation(Ref))
    return S;

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
              .bind(NodeID<Decl>::value))),
      Stm, Context);
  return findDeclMutation(Refs);
}

const Stmt *ExprMutationAnalyzer::findFunctionArgMutation(const Expr *Exp) {
  const auto NonConstRefParam = forEachArgumentWithParam(
      equalsNode(Exp),
      parmVarDecl(hasType(nonConstReferenceType())).bind("parm"));
  const auto IsInstantiated = hasDeclaration(isInstantiated());
  const auto FuncDecl = hasDeclaration(functionDecl().bind("func"));
  const auto Matches = match(
      traverse(
          ast_type_traits::TK_AsIs,
          findAll(
              expr(anyOf(callExpr(NonConstRefParam, IsInstantiated, FuncDecl,
                                  unless(callee(namedDecl(hasAnyName(
                                      "::std::move", "::std::forward"))))),
                         cxxConstructExpr(NonConstRefParam, IsInstantiated,
                                          FuncDecl)))
                  .bind(NodeID<Expr>::value))),
      Stm, Context);
  for (const auto &Nodes : Matches) {
    const auto *Exp = Nodes.getNodeAs<Expr>(NodeID<Expr>::value);
    const auto *Func = Nodes.getNodeAs<FunctionDecl>("func");
    if (!Func->getBody() || !Func->getPrimaryTemplate())
      return Exp;

    const auto *Parm = Nodes.getNodeAs<ParmVarDecl>("parm");
    const ArrayRef<ParmVarDecl *> AllParams =
        Func->getPrimaryTemplate()->getTemplatedDecl()->parameters();
    QualType ParmType =
        AllParams[std::min<size_t>(Parm->getFunctionScopeIndex(),
                                   AllParams.size() - 1)]
            ->getType();
    if (const auto *T = ParmType->getAs<PackExpansionType>())
      ParmType = T->getPattern();

    // If param type is forwarding reference, follow into the function
    // definition and see whether the param is mutated inside.
    if (const auto *RefType = ParmType->getAs<RValueReferenceType>()) {
      if (!RefType->getPointeeType().getQualifiers() &&
          RefType->getPointeeType()->getAs<TemplateTypeParmType>()) {
        std::unique_ptr<FunctionParmMutationAnalyzer> &Analyzer =
            FuncParmAnalyzer[Func];
        if (!Analyzer)
          Analyzer.reset(new FunctionParmMutationAnalyzer(*Func, Context));
        if (Analyzer->findMutation(Parm))
          return Exp;
        continue;
      }
    }
    // Not forwarding reference.
    return Exp;
  }
  return nullptr;
}

FunctionParmMutationAnalyzer::FunctionParmMutationAnalyzer(
    const FunctionDecl &Func, ASTContext &Context)
    : BodyAnalyzer(*Func.getBody(), Context) {
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(&Func)) {
    // CXXCtorInitializer might also mutate Param but they're not part of
    // function body, check them eagerly here since they're typically trivial.
    for (const CXXCtorInitializer *Init : Ctor->inits()) {
      ExprMutationAnalyzer InitAnalyzer(*Init->getInit(), Context);
      for (const ParmVarDecl *Parm : Ctor->parameters()) {
        if (Results.find(Parm) != Results.end())
          continue;
        if (const Stmt *S = InitAnalyzer.findMutation(Parm))
          Results[Parm] = S;
      }
    }
  }
}

const Stmt *
FunctionParmMutationAnalyzer::findMutation(const ParmVarDecl *Parm) {
  const auto Memoized = Results.find(Parm);
  if (Memoized != Results.end())
    return Memoized->second;

  if (const Stmt *S = BodyAnalyzer.findMutation(Parm))
    return Results[Parm] = S;

  return Results[Parm] = nullptr;
}

} // namespace clang
