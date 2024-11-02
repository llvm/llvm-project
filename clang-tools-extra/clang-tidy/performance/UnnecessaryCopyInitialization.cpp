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
static constexpr StringRef MethodDeclId = "methodDecl";
static constexpr StringRef FunctionDeclId = "functionDecl";
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

llvm::Optional<SourceLocation> firstLocAfterNewLine(SourceLocation Loc,
                                                    SourceManager &SM) {
  bool Invalid;
  const char *TextAfter = SM.getCharacterData(Loc, &Invalid);
  if (Invalid) {
    return llvm::None;
  }
  size_t Offset = std::strcspn(TextAfter, "\n");
  return Loc.getLocWithOffset(TextAfter[Offset] == '\0' ? Offset : Offset + 1);
}

void recordRemoval(const DeclStmt &Stmt, ASTContext &Context,
                   DiagnosticBuilder &Diagnostic) {
  auto &SM = Context.getSourceManager();
  // Attempt to remove trailing comments as well.
  auto Tok = utils::lexer::findNextTokenSkippingComments(Stmt.getEndLoc(), SM,
                                                         Context.getLangOpts());
  llvm::Optional<SourceLocation> PastNewLine =
      firstLocAfterNewLine(Stmt.getEndLoc(), SM);
  if (Tok && PastNewLine) {
    auto BeforeFirstTokenAfterComment = Tok->getLocation().getLocWithOffset(-1);
    // Remove until the end of the line or the end of a trailing comment which
    // ever comes first.
    auto End =
        SM.isBeforeInTranslationUnit(*PastNewLine, BeforeFirstTokenAfterComment)
            ? *PastNewLine
            : BeforeFirstTokenAfterComment;
    Diagnostic << FixItHint::CreateRemoval(
        SourceRange(Stmt.getBeginLoc(), End));
  } else {
    Diagnostic << FixItHint::CreateRemoval(Stmt.getSourceRange());
  }
}

AST_MATCHER_FUNCTION_P(StatementMatcher, isConstRefReturningMethodCall,
                       std::vector<StringRef>, ExcludedContainerTypes) {
  // Match method call expressions where the `this` argument is only used as
  // const, this will be checked in `check()` part. This returned const
  // reference is highly likely to outlive the local const reference of the
  // variable being declared. The assumption is that the const reference being
  // returned either points to a global static variable or to a member of the
  // called object.
  const auto MethodDecl =
      cxxMethodDecl(returns(hasCanonicalType(matchers::isReferenceToConst())))
          .bind(MethodDeclId);
  const auto ReceiverExpr = declRefExpr(to(varDecl().bind(ObjectArgId)));
  const auto ReceiverType =
      hasCanonicalType(recordType(hasDeclaration(namedDecl(
          unless(matchers::matchesAnyListedName(ExcludedContainerTypes))))));

  return expr(anyOf(
      cxxMemberCallExpr(callee(MethodDecl), on(ReceiverExpr),
                        thisPointerType(ReceiverType)),
      cxxOperatorCallExpr(callee(MethodDecl), hasArgument(0, ReceiverExpr),
                          hasArgument(0, hasType(ReceiverType)))));
}

AST_MATCHER_FUNCTION(StatementMatcher, isConstRefReturningFunctionCall) {
  // Only allow initialization of a const reference from a free function if it
  // has no arguments. Otherwise it could return an alias to one of its
  // arguments and the arguments need to be checked for const use as well.
  return callExpr(callee(functionDecl(returns(hasCanonicalType(
                                          matchers::isReferenceToConst())))
                             .bind(FunctionDeclId)),
                  argumentCountIs(0), unless(callee(cxxMethodDecl())))
      .bind(InitFunctionCallId);
}

AST_MATCHER_FUNCTION_P(StatementMatcher, initializerReturnsReferenceToConst,
                       std::vector<StringRef>, ExcludedContainerTypes) {
  auto OldVarDeclRef =
      declRefExpr(to(varDecl(hasLocalStorage()).bind(OldVarDeclId)));
  return expr(
      anyOf(isConstRefReturningFunctionCall(),
            isConstRefReturningMethodCall(ExcludedContainerTypes),
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
static bool isInitializingVariableImmutable(
    const VarDecl &InitializingVar, const Stmt &BlockStmt, ASTContext &Context,
    const std::vector<StringRef> &ExcludedContainerTypes) {
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

  auto Matches =
      match(initializerReturnsReferenceToConst(ExcludedContainerTypes),
            *InitializingVar.getInit(), Context);
  // The reference is initialized from a free function without arguments
  // returning a const reference. This is a global immutable object.
  if (selectFirst<CallExpr>(InitFunctionCallId, Matches) != nullptr)
    return true;
  // Check that the object argument is immutable as well.
  if (const auto *OrigVar = selectFirst<VarDecl>(ObjectArgId, Matches))
    return isInitializingVariableImmutable(*OrigVar, BlockStmt, Context,
                                           ExcludedContainerTypes);
  // Check that the old variable we reference is immutable as well.
  if (const auto *OrigVar = selectFirst<VarDecl>(OldVarDeclId, Matches))
    return isInitializingVariableImmutable(*OrigVar, BlockStmt, Context,
                                           ExcludedContainerTypes);

  return false;
}

bool isVariableUnused(const VarDecl &Var, const Stmt &BlockStmt,
                      ASTContext &Context) {
  return allDeclRefExprs(Var, BlockStmt, Context).empty();
}

const SubstTemplateTypeParmType *getSubstitutedType(const QualType &Type,
                                                    ASTContext &Context) {
  auto Matches = match(
      qualType(anyOf(substTemplateTypeParmType().bind("subst"),
                     hasDescendant(substTemplateTypeParmType().bind("subst")))),
      Type, Context);
  return selectFirst<SubstTemplateTypeParmType>("subst", Matches);
}

bool differentReplacedTemplateParams(const QualType &VarType,
                                     const QualType &InitializerType,
                                     ASTContext &Context) {
  if (const SubstTemplateTypeParmType *VarTmplType =
          getSubstitutedType(VarType, Context)) {
    if (const SubstTemplateTypeParmType *InitializerTmplType =
            getSubstitutedType(InitializerType, Context)) {
      return VarTmplType->getReplacedParameter()
                 ->desugar()
                 .getCanonicalType() !=
             InitializerTmplType->getReplacedParameter()
                 ->desugar()
                 .getCanonicalType();
    }
  }
  return false;
}

QualType constructorArgumentType(const VarDecl *OldVar,
                                 const BoundNodes &Nodes) {
  if (OldVar) {
    return OldVar->getType();
  }
  if (const auto *FuncDecl = Nodes.getNodeAs<FunctionDecl>(FunctionDeclId)) {
    return FuncDecl->getReturnType();
  }
  const auto *MethodDecl = Nodes.getNodeAs<CXXMethodDecl>(MethodDeclId);
  return MethodDecl->getReturnType();
}

} // namespace

UnnecessaryCopyInitialization::UnnecessaryCopyInitialization(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))),
      ExcludedContainerTypes(utils::options::parseStringList(
          Options.get("ExcludedContainerTypes", ""))) {}

void UnnecessaryCopyInitialization::registerMatchers(MatchFinder *Finder) {
  auto LocalVarCopiedFrom = [this](const internal::Matcher<Expr> &CopyCtorArg) {
    return compoundStmt(
               forEachDescendant(
                   declStmt(
                       unless(has(decompositionDecl())),
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
                                              isConstRefReturningMethodCall(
                                                  ExcludedContainerTypes))),
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

  // Don't apply the check if the variable and its initializer have different
  // replaced template parameter types. In this case the check triggers for a
  // template instantiation where the substituted types are the same, but
  // instantiations where the types differ and rely on implicit conversion would
  // no longer compile if we switched to a reference.
  if (differentReplacedTemplateParams(
          NewVar->getType(), constructorArgumentType(OldVar, Result.Nodes),
          *Result.Context))
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
      !isInitializingVariableImmutable(*ObjectArg, BlockStmt, Context,
                                       ExcludedContainerTypes))
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
      !isInitializingVariableImmutable(OldVar, BlockStmt, Context,
                                       ExcludedContainerTypes))
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
  Options.store(Opts, "ExcludedContainerTypes",
                utils::options::serializeStringList(ExcludedContainerTypes));
}

} // namespace performance
} // namespace tidy
} // namespace clang
