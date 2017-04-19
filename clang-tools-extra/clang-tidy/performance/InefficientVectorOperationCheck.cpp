//===--- InefficientVectorOperationCheck.cpp - clang-tidy------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InefficientVectorOperationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "../utils/DeclRefExprUtils.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

namespace {

// Matcher names. Given the code:
//
// \code
// void f() {
//   vector<T> v;
//   for (int i = 0; i < 10 + 1; ++i) {
//     v.push_back(i);
//   }
// }
// \endcode
//
// The matcher names are bound to following parts of the AST:
//   - LoopName: The entire for loop (as ForStmt).
//   - LoopParentName: The body of function f (as CompoundStmt).
//   - VectorVarDeclName: 'v' in  (as VarDecl).
//   - VectorVarDeclStmatName: The entire 'std::vector<T> v;' statement (as
//     DeclStmt).
//   - PushBackCallName: 'v.push_back(i)' (as cxxMemberCallExpr).
//   - LoopInitVarName: 'i' (as VarDecl).
//   - LoopEndExpr: '10+1' (as Expr).
static const char LoopCounterName[] = "for_loop_counter";
static const char LoopParentName[] = "loop_parent";
static const char VectorVarDeclName[] = "vector_var_decl";
static const char VectorVarDeclStmtName[] = "vector_var_decl_stmt";
static const char PushBackCallName[] = "push_back_call";

static const char LoopInitVarName[] = "loop_init_var";
static const char LoopEndExprName[] = "loop_end_expr";

} // namespace

void InefficientVectorOperationCheck::registerMatchers(MatchFinder *Finder) {
  const auto VectorDecl = cxxRecordDecl(hasName("::std::vector"));
  const auto VectorDefaultConstructorCall = cxxConstructExpr(
      hasType(VectorDecl),
      hasDeclaration(cxxConstructorDecl(isDefaultConstructor())));
  const auto VectorVarDecl =
      varDecl(hasInitializer(VectorDefaultConstructorCall))
          .bind(VectorVarDeclName);
  const auto PushBackCallExpr =
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("push_back"))), on(hasType(VectorDecl)),
          onImplicitObjectArgument(declRefExpr(to(VectorVarDecl))))
          .bind(PushBackCallName);
  const auto PushBackCall =
      expr(anyOf(PushBackCallExpr, exprWithCleanups(has(PushBackCallExpr))));
  const auto VectorVarDefStmt =
      declStmt(hasSingleDecl(equalsBoundNode(VectorVarDeclName)))
          .bind(VectorVarDeclStmtName);

  const auto LoopVarInit =
      declStmt(hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0))))
                                 .bind(LoopInitVarName)));
  const auto RefersToLoopVar = ignoringParenImpCasts(
      declRefExpr(to(varDecl(equalsBoundNode(LoopInitVarName)))));

  // Match counter-based for loops:
  //  for (int i = 0; i < n; ++i) { v.push_back(...); }
  //
  // FIXME: Support more types of counter-based loops like decrement loops.
  Finder->addMatcher(
      forStmt(
          hasLoopInit(LoopVarInit),
          hasCondition(binaryOperator(
              hasOperatorName("<"), hasLHS(RefersToLoopVar),
              hasRHS(expr(unless(hasDescendant(expr(RefersToLoopVar))))
                         .bind(LoopEndExprName)))),
          hasIncrement(unaryOperator(hasOperatorName("++"),
                                     hasUnaryOperand(RefersToLoopVar))),
          hasBody(anyOf(compoundStmt(statementCountIs(1), has(PushBackCall)),
                        PushBackCall)),
          hasParent(compoundStmt(has(VectorVarDefStmt)).bind(LoopParentName)))
          .bind(LoopCounterName),
      this);
}

void InefficientVectorOperationCheck::check(
    const MatchFinder::MatchResult &Result) {
  auto* Context = Result.Context;
  if (Context->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const auto *ForLoop = Result.Nodes.getNodeAs<ForStmt>(LoopCounterName);
  const auto *PushBackCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>(PushBackCallName);
  const auto *LoopEndExpr = Result.Nodes.getNodeAs<Expr>(LoopEndExprName);
  const auto *LoopParent = Result.Nodes.getNodeAs<CompoundStmt>(LoopParentName);
  const auto *VectorVarDecl =
      Result.Nodes.getNodeAs<VarDecl>(VectorVarDeclName);

  llvm::SmallPtrSet<const DeclRefExpr *, 16> AllVectorVarRefs =
      utils::decl_ref_expr::allDeclRefExprs(*VectorVarDecl, *LoopParent,
                                            *Context);
  for (const auto *Ref : AllVectorVarRefs) {
    // Skip cases where there are usages (defined as DeclRefExpr that refers to
    // "v") of vector variable `v` before the for loop. We consider these usages
    // are operations causing memory preallocation (e.g. "v.resize(n)",
    // "v.reserve(n)").
    //
    // FIXME: make it more intelligent to identify the pre-allocating operations
    // before the for loop.
    if (SM.isBeforeInTranslationUnit(Ref->getLocation(),
                                     ForLoop->getLocStart())) {
      return;
    }
  }

  llvm::StringRef LoopEndSource = Lexer::getSourceText(
      CharSourceRange::getTokenRange(LoopEndExpr->getSourceRange()), SM,
      Context->getLangOpts());
  llvm::StringRef VectorVarName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          PushBackCall->getImplicitObjectArgument()->getSourceRange()),
      SM, Context->getLangOpts());
  std::string ReserveStmt =
      (VectorVarName + ".reserve(" + LoopEndSource + ");\n").str();

  diag(PushBackCall->getLocStart(),
       "'push_back' is called inside a loop; "
       "consider pre-allocating the vector capacity before the loop")
      << FixItHint::CreateInsertion(ForLoop->getLocStart(), ReserveStmt);
}

} // namespace performance
} // namespace tidy
} // namespace clang
