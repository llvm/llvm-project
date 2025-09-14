//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveConstArgCheck.h"

#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static void replaceCallWithArg(const CallExpr *Call, DiagnosticBuilder &Diag,
                               const SourceManager &SM,
                               const LangOptions &LangOpts) {
  const Expr *Arg = Call->getArg(0);

  CharSourceRange BeforeArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getBeginLoc(), Arg->getBeginLoc()),
      SM, LangOpts);
  CharSourceRange AfterArgumentsRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Call->getEndLoc(),
                                    Call->getEndLoc().getLocWithOffset(1)),
      SM, LangOpts);

  if (BeforeArgumentsRange.isValid() && AfterArgumentsRange.isValid()) {
    Diag << FixItHint::CreateRemoval(BeforeArgumentsRange)
         << FixItHint::CreateRemoval(AfterArgumentsRange);
  }
}

void MoveConstArgCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckTriviallyCopyableMove", CheckTriviallyCopyableMove);
  Options.store(Opts, "CheckMoveToConstRef", CheckMoveToConstRef);
}

void MoveConstArgCheck::registerMatchers(MatchFinder *Finder) {
  auto MoveCallMatcher =
      callExpr(callee(functionDecl(hasName("::std::move"))), argumentCountIs(1),
               unless(isInTemplateInstantiation()))
          .bind("call-move");

  // Match ternary expressions where either branch contains std::move
  auto TernaryWithMoveMatcher =
      conditionalOperator(hasDescendant(MoveCallMatcher));

  Finder->addMatcher(
      expr(anyOf(
          castExpr(hasSourceExpression(MoveCallMatcher)),
          cxxConstructExpr(hasDeclaration(cxxConstructorDecl(anyOf(
                               isCopyConstructor(), isMoveConstructor()))),
                           hasArgument(0, MoveCallMatcher)))),
      this);

  auto ConstTypeParmMatcher =
      qualType(references(isConstQualified())).bind("invocation-parm-type");
  auto RValueTypeParmMatcher =
      qualType(rValueReferenceType()).bind("invocation-parm-type");
  // Matches respective ParmVarDecl for a CallExpr or CXXConstructExpr.
  auto ArgumentWithParamMatcher = forEachArgumentWithParam(
      anyOf(MoveCallMatcher, TernaryWithMoveMatcher),
      parmVarDecl(
          anyOf(hasType(ConstTypeParmMatcher), hasType(RValueTypeParmMatcher)))
          .bind("invocation-parm"));
  // Matches respective types of arguments for a CallExpr or CXXConstructExpr
  // and it works on calls through function pointers as well.
  auto ArgumentWithParamTypeMatcher = forEachArgumentWithParamType(
      anyOf(MoveCallMatcher, TernaryWithMoveMatcher),
      anyOf(ConstTypeParmMatcher, RValueTypeParmMatcher));

  Finder->addMatcher(
      invocation(anyOf(ArgumentWithParamMatcher, ArgumentWithParamTypeMatcher))
          .bind("receiving-expr"),
      this);
}

static bool isRValueReferenceParam(const Expr *Invocation,
                                   const QualType *InvocationParmType,
                                   const Expr *Arg) {
  if (Invocation && (*InvocationParmType)->isRValueReferenceType() &&
      Arg->isLValue()) {
    if (!Invocation->getType()->isRecordType())
      return true;
    if (const auto *ConstructCallExpr =
            dyn_cast<CXXConstructExpr>(Invocation)) {
      if (const auto *ConstructorDecl = ConstructCallExpr->getConstructor()) {
        if (!ConstructorDecl->isCopyOrMoveConstructor() &&
            !ConstructorDecl->isDefaultConstructor())
          return true;
      }
    }
  }
  return false;
}

void MoveConstArgCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CallMove = Result.Nodes.getNodeAs<CallExpr>("call-move");
  const auto *ReceivingExpr = Result.Nodes.getNodeAs<Expr>("receiving-expr");
  const auto *InvocationParm =
      Result.Nodes.getNodeAs<ParmVarDecl>("invocation-parm");
  const auto *InvocationParmType =
      Result.Nodes.getNodeAs<QualType>("invocation-parm-type");

  // Skipping matchers which have been matched.
  if (!ReceivingExpr && AlreadyCheckedMoves.contains(CallMove))
    return;

  if (ReceivingExpr)
    AlreadyCheckedMoves.insert(CallMove);

  const Expr *Arg = CallMove->getArg(0);
  const QualType ArgType = Arg->getType().getCanonicalType();
  SourceManager &SM = Result.Context->getSourceManager();

  CharSourceRange MoveRange =
      CharSourceRange::getCharRange(CallMove->getSourceRange());
  CharSourceRange FileMoveRange =
      Lexer::makeFileCharRange(MoveRange, SM, getLangOpts());
  if (!FileMoveRange.isValid())
    return;

  bool IsConstArg = ArgType.isConstQualified();
  bool IsTriviallyCopyable = ArgType.isTriviallyCopyableType(*Result.Context);

  if (IsConstArg || IsTriviallyCopyable) {
    if (const CXXRecordDecl *R = ArgType->getAsCXXRecordDecl()) {
      // According to [expr.prim.lambda]p3, "whether the closure type is
      // trivially copyable" property can be changed by the implementation of
      // the language, so we shouldn't rely on it when issuing diagnostics.
      if (R->isLambda())
        return;
      // Don't warn when the type is not copyable.
      for (const auto *Ctor : R->ctors()) {
        if (Ctor->isCopyConstructor() && Ctor->isDeleted())
          return;
      }
    }

    if (!IsConstArg && IsTriviallyCopyable && !CheckTriviallyCopyableMove)
      return;

    bool IsVariable = isa<DeclRefExpr>(Arg);
    // std::move shouldn't be removed when an lvalue wrapped by std::move is
    // passed to the function with an rvalue reference parameter.
    bool IsRVRefParam =
        isRValueReferenceParam(ReceivingExpr, InvocationParmType, Arg);
    const auto *Var =
        IsVariable ? dyn_cast<DeclRefExpr>(Arg)->getDecl() : nullptr;

    {
      auto Diag = diag(FileMoveRange.getBegin(),
                       "std::move of the %select{|const }0"
                       "%select{expression|variable %5}1 "
                       "%select{|of the trivially-copyable type %6 }2"
                       "has no effect%select{; remove std::move()|}3"
                       "%select{| or make the variable non-const}4")
                  << IsConstArg << IsVariable << IsTriviallyCopyable
                  << IsRVRefParam
                  << (IsConstArg && IsVariable && !IsTriviallyCopyable) << Var
                  << Arg->getType();
      if (!IsRVRefParam)
        replaceCallWithArg(CallMove, Diag, SM, getLangOpts());
    }
    if (IsRVRefParam) {
      // Generate notes for an invocation with an rvalue reference parameter.
      const auto *ReceivingCallExpr = dyn_cast<CallExpr>(ReceivingExpr);
      const auto *ReceivingConstructExpr =
          dyn_cast<CXXConstructExpr>(ReceivingExpr);
      // Skipping the invocation which is a template instantiation.
      if ((!ReceivingCallExpr || !ReceivingCallExpr->getDirectCallee() ||
           ReceivingCallExpr->getDirectCallee()->isTemplateInstantiation()) &&
          (!ReceivingConstructExpr ||
           !ReceivingConstructExpr->getConstructor() ||
           ReceivingConstructExpr->getConstructor()->isTemplateInstantiation()))
        return;

      const NamedDecl *FunctionName = nullptr;
      FunctionName =
          ReceivingCallExpr
              ? ReceivingCallExpr->getDirectCallee()->getUnderlyingDecl()
              : ReceivingConstructExpr->getConstructor()->getUnderlyingDecl();

      QualType NoRefType = (*InvocationParmType)->getPointeeType();
      PrintingPolicy PolicyWithSuppressedTag(getLangOpts());
      PolicyWithSuppressedTag.SuppressTagKeyword = true;
      PolicyWithSuppressedTag.SuppressUnwrittenScope = true;
      std::string ExpectParmTypeName =
          NoRefType.getAsString(PolicyWithSuppressedTag);
      if (!NoRefType->isPointerType()) {
        NoRefType.addConst();
        ExpectParmTypeName =
            NoRefType.getAsString(PolicyWithSuppressedTag) + " &";
      }

      diag(InvocationParm->getLocation(),
           "consider changing the %ordinal0 parameter of %1 from %2 to '%3'",
           DiagnosticIDs::Note)
          << (InvocationParm->getFunctionScopeIndex() + 1) << FunctionName
          << *InvocationParmType << ExpectParmTypeName;
    }
  } else if (ReceivingExpr && CheckMoveToConstRef) {
    if ((*InvocationParmType)->isRValueReferenceType())
      return;

    {
      auto Diag = diag(FileMoveRange.getBegin(),
                       "passing result of std::move() as a const reference "
                       "argument; no move will actually happen");

      replaceCallWithArg(CallMove, Diag, SM, getLangOpts());
    }

    if (const CXXRecordDecl *RecordDecl = ArgType->getAsCXXRecordDecl();
        RecordDecl && RecordDecl->hasDefinition() &&
        !(RecordDecl->hasMoveConstructor() &&
          RecordDecl->hasMoveAssignment())) {
      const bool MissingMoveAssignment = !RecordDecl->hasMoveAssignment();
      const bool MissingMoveConstructor = !RecordDecl->hasMoveConstructor();
      const bool MissingBoth = MissingMoveAssignment && MissingMoveConstructor;

      diag(RecordDecl->getLocation(),
           "%0 is not move "
           "%select{|assignable}1%select{|/}2%select{|constructible}3",
           DiagnosticIDs::Note)
          << RecordDecl << MissingMoveAssignment << MissingBoth
          << MissingMoveConstructor;
    }
  }
}

} // namespace clang::tidy::performance
