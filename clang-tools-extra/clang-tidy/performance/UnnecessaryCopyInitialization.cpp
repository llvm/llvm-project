//===--- UnnecessaryCopyInitialization.cpp - clang-tidy--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryCopyInitialization.h"
#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"

namespace clang {
namespace tidy {
namespace performance {
namespace {

using namespace ::clang::ast_matchers;
using llvm::StringRef;
using utils::decl_ref_expr::allDeclRefExprs;
using utils::decl_ref_expr::isOnlyUsedAsConst;

static constexpr StringRef ObjectArgId = "objectArg";
static constexpr StringRef InitFunctionCallId = "initFunctionCall";
static constexpr StringRef OldVarDeclId = "oldVarDecl";

void recordFixes(const VarDecl &Var, ASTContext &Context,
                 DiagnosticBuilder &Diagnostic) {
  Diagnostic << utils::fixit::changeVarDeclToReference(Var, Context);
  if (!Var.getType().isLocalConstQualified()) {
    if (llvm::Optional<FixItHint> Fix = utils::fixit::addQualifierToVarDecl(
            Var, Context, DeclSpec::TQ::TQ_const))
      Diagnostic << *Fix;
  }
}

void recordRemoval(const DeclStmt &Stmt, ASTContext &Context,
                   DiagnosticBuilder &Diagnostic) {
  // Attempt to remove the whole line until the next non-comment token.
  auto Tok = utils::lexer::findNextTokenSkippingComments(
      Stmt.getEndLoc(), Context.getSourceManager(), Context.getLangOpts());
  if (Tok) {
    Diagnostic << FixItHint::CreateRemoval(SourceRange(
        Stmt.getBeginLoc(), Tok->getLocation().getLocWithOffset(-1)));
  } else {
    Diagnostic << FixItHint::CreateRemoval(Stmt.getSourceRange());
  }
}

AST_MATCHER_FUNCTION(StatementMatcher, isConstRefReturningMethodCall) {
  // Match method call expressions where the `this` argument is only used as
  // const, this will be checked in `check()` part. This returned const
  // reference is highly likely to outlive the local const reference of the
  // variable being declared. The assumption is that the const reference being
  // returned either points to a global static variable or to a member of the
  // called object.
  return cxxMemberCallExpr(
      callee(cxxMethodDecl(returns(matchers::isReferenceToConst()))),
      on(declRefExpr(to(varDecl().bind(ObjectArgId)))));
}

AST_MATCHER_FUNCTION(StatementMatcher, isConstRefReturningFunctionCall) {
  // Only allow initialization of a const reference from a free function if it
  // has no arguments. Otherwise it could return an alias to one of its
  // arguments and the arguments need to be checked for const use as well.
  return callExpr(callee(functionDecl(returns(matchers::isReferenceToConst()))),
                  argumentCountIs(0), unless(callee(cxxMethodDecl())))
      .bind(InitFunctionCallId);
}

AST_MATCHER_FUNCTION(StatementMatcher, initializerReturnsReferenceToConst) {
  auto OldVarDeclRef =
      declRefExpr(to(varDecl(hasLocalStorage()).bind(OldVarDeclId)));
  return expr(
      anyOf(isConstRefReturningFunctionCall(), isConstRefReturningMethodCall(),
            ignoringImpCasts(OldVarDeclRef),
            ignoringImpCasts(unaryOperator(hasOperatorName("&"),
                                           hasUnaryOperand(OldVarDeclRef)))));
}

// This checks that the variable itself is only used as const, and also makes
// sure that it does not reference another variable that could be modified in
// the BlockStmt. It does this by checking the following:
// 1. If the variable is neither a reference nor a pointer then the
// isOnlyUsedAsConst() check is sufficient.
// 2. If the (reference or pointer) variable is not initialized in a DeclStmt in
// the BlockStmt. In this case its pointee is likely not modified (unless it
// is passed as an alias into the method as well).
// 3. If the reference is initialized from a reference to const. This is
// the same set of criteria we apply when identifying the unnecessary copied
// variable in this check to begin with. In this case we check whether the
// object arg or variable that is referenced is immutable as well.
static bool isInitializingVariableImmutable(const VarDecl &InitializingVar,
                                            const Stmt &BlockStmt,
                                            ASTContext &Context) {
  if (!isOnlyUsedAsConst(InitializingVar, BlockStmt, Context))
    return false;

  QualType T = InitializingVar.getType().getCanonicalType();
  // The variable is a value type and we know it is only used as const. Safe
  // to reference it and avoid the copy.
  if (!isa<ReferenceType, PointerType>(T))
    return true;

  // The reference or pointer is not declared and hence not initialized anywhere
  // in the function. We assume its pointee is not modified then.
  if (!InitializingVar.isLocalVarDecl() || !InitializingVar.hasInit()) {
    return true;
  }

  auto Matches = match(initializerReturnsReferenceToConst(),
                       *InitializingVar.getInit(), Context);
  // The reference is initialized from a free function without arguments
  // returning a const reference. This is a global immutable object.
  if (selectFirst<CallExpr>(InitFunctionCallId, Matches) != nullptr)
    return true;
  // Check that the object argument is immutable as well.
  if (const auto *OrigVar = selectFirst<VarDecl>(ObjectArgId, Matches))
    return isInitializingVariableImmutable(*OrigVar, BlockStmt, Context);
  // Check that the old variable we reference is immutable as well.
  if (const auto *OrigVar = selectFirst<VarDecl>(OldVarDeclId, Matches))
    return isInitializingVariableImmutable(*OrigVar, BlockStmt, Context);

  return false;
}

bool isVariableUnused(const VarDecl &Var, const Stmt &BlockStmt,
                      ASTContext &Context) {
  return allDeclRefExprs(Var, BlockStmt, Context).empty();
}

} // namespace

UnnecessaryCopyInitialization::UnnecessaryCopyInitialization(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

void UnnecessaryCopyInitialization::registerMatchers(MatchFinder *Finder) {
  auto LocalVarCopiedFrom = [this](const internal::Matcher<Expr> &CopyCtorArg) {
    return compoundStmt(
               forEachDescendant(
                   declStmt(
                       has(varDecl(hasLocalStorage(),
                                   hasType(qualType(
                                       hasCanonicalType(allOf(
                                           matchers::isExpensiveToCopy(),
                                           unless(hasDeclaration(namedDecl(
                                               hasName("::std::function")))))),
                                       unless(hasDeclaration(namedDecl(
                                           matchers::matchesAnyListedName(
                                               AllowedTypes)))))),
                                   unless(isImplicit()),
                                   hasInitializer(traverse(
                                       TK_AsIs,
                                       cxxConstructExpr(
                                           hasDeclaration(cxxConstructorDecl(
                                               isCopyConstructor())),
                                           hasArgument(0, CopyCtorArg))
                                           .bind("ctorCall"))))
                               .bind("newVarDecl")))
                       .bind("declStmt")))
        .bind("blockStmt");
  };

  Finder->addMatcher(LocalVarCopiedFrom(anyOf(isConstRefReturningFunctionCall(),
                                              isConstRefReturningMethodCall())),
                     this);

  Finder->addMatcher(LocalVarCopiedFrom(declRefExpr(
                         to(varDecl(hasLocalStorage()).bind(OldVarDeclId)))),
                     this);
}

void UnnecessaryCopyInitialization::check(
    const MatchFinder::MatchResult &Result) {
  const auto *NewVar = Result.Nodes.getNodeAs<VarDecl>("newVarDecl");
  const auto *OldVar = Result.Nodes.getNodeAs<VarDecl>(OldVarDeclId);
  const auto *ObjectArg = Result.Nodes.getNodeAs<VarDecl>(ObjectArgId);
  const auto *BlockStmt = Result.Nodes.getNodeAs<Stmt>("blockStmt");
  const auto *CtorCall = Result.Nodes.getNodeAs<CXXConstructExpr>("ctorCall");
  const auto *Stmt = Result.Nodes.getNodeAs<DeclStmt>("declStmt");

  TraversalKindScope RAII(*Result.Context, TK_AsIs);

  // Do not propose fixes if the DeclStmt has multiple VarDecls or in macros
  // since we cannot place them correctly.
  bool IssueFix = Stmt->isSingleDecl() && !NewVar->getLocation().isMacroID();

  // A constructor that looks like T(const T& t, bool arg = false) counts as a
  // copy only when it is called with default arguments for the arguments after
  // the first.
  for (unsigned int I = 1; I < CtorCall->getNumArgs(); ++I)
    if (!CtorCall->getArg(I)->isDefaultArgument())
      return;

  if (OldVar == nullptr) {
    handleCopyFromMethodReturn(*NewVar, *BlockStmt, *Stmt, IssueFix, ObjectArg,
                               *Result.Context);
  } else {
    handleCopyFromLocalVar(*NewVar, *OldVar, *BlockStmt, *Stmt, IssueFix,
                           *Result.Context);
  }
}

void UnnecessaryCopyInitialization::handleCopyFromMethodReturn(
    const VarDecl &Var, const Stmt &BlockStmt, const DeclStmt &Stmt,
    bool IssueFix, const VarDecl *ObjectArg, ASTContext &Context) {
  bool IsConstQualified = Var.getType().isConstQualified();
  if (!IsConstQualified && !isOnlyUsedAsConst(Var, BlockStmt, Context))
    return;
  if (ObjectArg != nullptr &&
      !isInitializingVariableImmutable(*ObjectArg, BlockStmt, Context))
    return;
  if (isVariableUnused(Var, BlockStmt, Context)) {
    auto Diagnostic =
        diag(Var.getLocation(),
             "the %select{|const qualified }0variable %1 is copy-constructed "
             "from a const reference but is never used; consider "
             "removing the statement")
        << IsConstQualified << &Var;
    if (IssueFix)
      recordRemoval(Stmt, Context, Diagnostic);
  } else {
    auto Diagnostic =
        diag(Var.getLocation(),
             "the %select{|const qualified }0variable %1 is copy-constructed "
             "from a const reference%select{ but is only used as const "
             "reference|}0; consider making it a const reference")
        << IsConstQualified << &Var;
    if (IssueFix)
      recordFixes(Var, Context, Diagnostic);
  }
}

void UnnecessaryCopyInitialization::handleCopyFromLocalVar(
    const VarDecl &NewVar, const VarDecl &OldVar, const Stmt &BlockStmt,
    const DeclStmt &Stmt, bool IssueFix, ASTContext &Context) {
  if (!isOnlyUsedAsConst(NewVar, BlockStmt, Context) ||
      !isInitializingVariableImmutable(OldVar, BlockStmt, Context))
    return;

  if (isVariableUnused(NewVar, BlockStmt, Context)) {
    auto Diagnostic = diag(NewVar.getLocation(),
                           "local copy %0 of the variable %1 is never modified "
                           "and never used; "
                           "consider removing the statement")
                      << &NewVar << &OldVar;
    if (IssueFix)
      recordRemoval(Stmt, Context, Diagnostic);
  } else {
    auto Diagnostic =
        diag(NewVar.getLocation(),
             "local copy %0 of the variable %1 is never modified; "
             "consider avoiding the copy")
        << &NewVar << &OldVar;
    if (IssueFix)
      recordFixes(NewVar, Context, Diagnostic);
  }
}

void UnnecessaryCopyInitialization::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

} // namespace performance
} // namespace tidy
} // namespace clang
