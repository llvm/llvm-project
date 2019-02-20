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
#include "../utils/OptionsUtils.h"

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
//   - LoopCounterName: The entire for loop (as ForStmt).
//   - LoopParentName: The body of function f (as CompoundStmt).
//   - VectorVarDeclName: 'v' in  (as VarDecl).
//   - VectorVarDeclStmatName: The entire 'std::vector<T> v;' statement (as
//     DeclStmt).
//   - PushBackOrEmplaceBackCallName: 'v.push_back(i)' (as cxxMemberCallExpr).
//   - LoopInitVarName: 'i' (as VarDecl).
//   - LoopEndExpr: '10+1' (as Expr).
static const char LoopCounterName[] = "for_loop_counter";
static const char LoopParentName[] = "loop_parent";
static const char VectorVarDeclName[] = "vector_var_decl";
static const char VectorVarDeclStmtName[] = "vector_var_decl_stmt";
static const char PushBackOrEmplaceBackCallName[] = "append_call";
static const char LoopInitVarName[] = "loop_init_var";
static const char LoopEndExprName[] = "loop_end_expr";

static const char RangeLoopName[] = "for_range_loop";

ast_matchers::internal::Matcher<Expr> supportedContainerTypesMatcher() {
  return hasType(cxxRecordDecl(hasAnyName(
      "::std::vector", "::std::set", "::std::unordered_set", "::std::map",
      "::std::unordered_map", "::std::array", "::std::deque")));
}

} // namespace

InefficientVectorOperationCheck::InefficientVectorOperationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      VectorLikeClasses(utils::options::parseStringList(
          Options.get("VectorLikeClasses", "::std::vector"))) {}

void InefficientVectorOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "VectorLikeClasses",
                utils::options::serializeStringList(VectorLikeClasses));
}

void InefficientVectorOperationCheck::registerMatchers(MatchFinder *Finder) {
  const auto VectorDecl = cxxRecordDecl(hasAnyName(SmallVector<StringRef, 5>(
      VectorLikeClasses.begin(), VectorLikeClasses.end())));
  const auto VectorDefaultConstructorCall = cxxConstructExpr(
      hasType(VectorDecl),
      hasDeclaration(cxxConstructorDecl(isDefaultConstructor())));
  const auto VectorVarDecl =
      varDecl(hasInitializer(VectorDefaultConstructorCall))
          .bind(VectorVarDeclName);
  const auto VectorAppendCallExpr =
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasAnyName("push_back", "emplace_back"))),
          on(hasType(VectorDecl)),
          onImplicitObjectArgument(declRefExpr(to(VectorVarDecl))))
          .bind(PushBackOrEmplaceBackCallName);
  const auto VectorAppendCall = expr(ignoringImplicit(VectorAppendCallExpr));
  const auto VectorVarDefStmt =
      declStmt(hasSingleDecl(equalsBoundNode(VectorVarDeclName)))
          .bind(VectorVarDeclStmtName);

  const auto LoopVarInit =
      declStmt(hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0))))
                                 .bind(LoopInitVarName)));
  const auto RefersToLoopVar = ignoringParenImpCasts(
      declRefExpr(to(varDecl(equalsBoundNode(LoopInitVarName)))));

  // Matchers for the loop whose body has only 1 push_back/emplace_back calling
  // statement.
  const auto HasInterestingLoopBody =
      hasBody(anyOf(compoundStmt(statementCountIs(1), has(VectorAppendCall)),
                    VectorAppendCall));
  const auto InInterestingCompoundStmt =
      hasParent(compoundStmt(has(VectorVarDefStmt)).bind(LoopParentName));

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
          HasInterestingLoopBody, InInterestingCompoundStmt)
          .bind(LoopCounterName),
      this);

  // Match for-range loops:
  //   for (const auto& E : data) { v.push_back(...); }
  //
  // FIXME: Support more complex range-expressions.
  Finder->addMatcher(
      cxxForRangeStmt(
          hasRangeInit(declRefExpr(supportedContainerTypesMatcher())),
          HasInterestingLoopBody, InInterestingCompoundStmt)
          .bind(RangeLoopName),
      this);
}

void InefficientVectorOperationCheck::check(
    const MatchFinder::MatchResult &Result) {
  auto* Context = Result.Context;
  if (Context->getDiagnostics().hasUncompilableErrorOccurred())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const auto *VectorVarDecl =
      Result.Nodes.getNodeAs<VarDecl>(VectorVarDeclName);
  const auto *ForLoop = Result.Nodes.getNodeAs<ForStmt>(LoopCounterName);
  const auto *RangeLoop =
      Result.Nodes.getNodeAs<CXXForRangeStmt>(RangeLoopName);
  const auto *VectorAppendCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>(PushBackOrEmplaceBackCallName);
  const auto *LoopEndExpr = Result.Nodes.getNodeAs<Expr>(LoopEndExprName);
  const auto *LoopParent = Result.Nodes.getNodeAs<CompoundStmt>(LoopParentName);

  const Stmt *LoopStmt = ForLoop;
  if (!LoopStmt)
    LoopStmt = RangeLoop;

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
                                     LoopStmt->getBeginLoc())) {
      return;
    }
  }

  llvm::StringRef VectorVarName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          VectorAppendCall->getImplicitObjectArgument()->getSourceRange()),
      SM, Context->getLangOpts());

  std::string ReserveStmt;
  // Handle for-range loop cases.
  if (RangeLoop) {
    // Get the range-expression in a for-range statement represented as
    // `for (range-declarator: range-expression)`.
    StringRef RangeInitExpName = Lexer::getSourceText(
        CharSourceRange::getTokenRange(
            RangeLoop->getRangeInit()->getSourceRange()),
        SM, Context->getLangOpts());

    ReserveStmt =
        (VectorVarName + ".reserve(" + RangeInitExpName + ".size()" + ");\n")
            .str();
  } else if (ForLoop) {
    // Handle counter-based loop cases.
    StringRef LoopEndSource = Lexer::getSourceText(
        CharSourceRange::getTokenRange(LoopEndExpr->getSourceRange()), SM,
        Context->getLangOpts());
    ReserveStmt = (VectorVarName + ".reserve(" + LoopEndSource + ");\n").str();
  }

  auto Diag =
      diag(VectorAppendCall->getBeginLoc(),
           "%0 is called inside a loop; "
           "consider pre-allocating the vector capacity before the loop")
      << VectorAppendCall->getMethodDecl()->getDeclName();

  if (!ReserveStmt.empty())
    Diag << FixItHint::CreateInsertion(LoopStmt->getBeginLoc(), ReserveStmt);
}

} // namespace performance
} // namespace tidy
} // namespace clang
