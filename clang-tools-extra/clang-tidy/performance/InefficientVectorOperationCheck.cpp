//===--- InefficientVectorOperationCheck.cpp - clang-tidy------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
//
//   SomeProto p;
//   for (int i = 0; i < 10 + 1; ++i) {
//     p.add_xxx(i);
//   }
// }
// \endcode
//
// The matcher names are bound to following parts of the AST:
//   - LoopCounterName: The entire for loop (as ForStmt).
//   - LoopParentName: The body of function f (as CompoundStmt).
//   - VectorVarDeclName: 'v' (as VarDecl).
//   - VectorVarDeclStmatName: The entire 'std::vector<T> v;' statement (as
//     DeclStmt).
//   - PushBackOrEmplaceBackCallName: 'v.push_back(i)' (as cxxMemberCallExpr).
//   - LoopInitVarName: 'i' (as VarDecl).
//   - LoopEndExpr: '10+1' (as Expr).
// If EnableProto, the proto related names are bound to the following parts:
//   - ProtoVarDeclName: 'p' (as VarDecl).
//   - ProtoVarDeclStmtName: The entire 'SomeProto p;' statement (as DeclStmt).
//   - ProtoAddFieldCallName: 'p.add_xxx(i)' (as cxxMemberCallExpr).
static const char LoopCounterName[] = "for_loop_counter";
static const char LoopParentName[] = "loop_parent";
static const char VectorVarDeclName[] = "vector_var_decl";
static const char VectorVarDeclStmtName[] = "vector_var_decl_stmt";
static const char PushBackOrEmplaceBackCallName[] = "append_call";
static const char ProtoVarDeclName[] = "proto_var_decl";
static const char ProtoVarDeclStmtName[] = "proto_var_decl_stmt";
static const char ProtoAddFieldCallName[] = "proto_add_field";
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
          Options.get("VectorLikeClasses", "::std::vector"))),
      EnableProto(Options.getLocalOrGlobal("EnableProto", false)) {}

void InefficientVectorOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "VectorLikeClasses",
                utils::options::serializeStringList(VectorLikeClasses));
  Options.store(Opts, "EnableProto", EnableProto);
}

void InefficientVectorOperationCheck::AddMatcher(
    const DeclarationMatcher &TargetRecordDecl, StringRef VarDeclName,
    StringRef VarDeclStmtName, const DeclarationMatcher &AppendMethodDecl,
    StringRef AppendCallName, MatchFinder *Finder) {
  const auto DefaultConstructorCall = cxxConstructExpr(
      hasType(TargetRecordDecl),
      hasDeclaration(cxxConstructorDecl(isDefaultConstructor())));
  const auto TargetVarDecl =
      varDecl(hasInitializer(DefaultConstructorCall)).bind(VarDeclName);
  const auto TargetVarDefStmt =
      declStmt(hasSingleDecl(equalsBoundNode(VarDeclName)))
          .bind(VarDeclStmtName);

  const auto AppendCallExpr =
      cxxMemberCallExpr(
          callee(AppendMethodDecl), on(hasType(TargetRecordDecl)),
          onImplicitObjectArgument(declRefExpr(to(TargetVarDecl))))
          .bind(AppendCallName);
  const auto AppendCall = expr(ignoringImplicit(AppendCallExpr));
  const auto LoopVarInit =
      declStmt(hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0))))
                                 .bind(LoopInitVarName)));
  const auto RefersToLoopVar = ignoringParenImpCasts(
      declRefExpr(to(varDecl(equalsBoundNode(LoopInitVarName)))));

  // Matchers for the loop whose body has only 1 push_back/emplace_back calling
  // statement.
  const auto HasInterestingLoopBody = hasBody(
      anyOf(compoundStmt(statementCountIs(1), has(AppendCall)), AppendCall));
  const auto InInterestingCompoundStmt =
      hasParent(compoundStmt(has(TargetVarDefStmt)).bind(LoopParentName));

  // Match counter-based for loops:
  //  for (int i = 0; i < n; ++i) {
  //    v.push_back(...);
  //    // Or: proto.add_xxx(...);
  //  }
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
  //   for (const auto& E : data) {
  //     v.push_back(...);
  //     // Or: proto.add_xxx(...);
  //   }
  //
  // FIXME: Support more complex range-expressions.
  Finder->addMatcher(
      cxxForRangeStmt(
          hasRangeInit(declRefExpr(supportedContainerTypesMatcher())),
          HasInterestingLoopBody, InInterestingCompoundStmt)
          .bind(RangeLoopName),
      this);
}

void InefficientVectorOperationCheck::registerMatchers(MatchFinder *Finder) {
  const auto VectorDecl = cxxRecordDecl(hasAnyName(SmallVector<StringRef, 5>(
      VectorLikeClasses.begin(), VectorLikeClasses.end())));
  const auto AppendMethodDecl =
      cxxMethodDecl(hasAnyName("push_back", "emplace_back"));
  AddMatcher(VectorDecl, VectorVarDeclName, VectorVarDeclStmtName,
             AppendMethodDecl, PushBackOrEmplaceBackCallName, Finder);

  if (EnableProto) {
    const auto ProtoDecl =
        cxxRecordDecl(isDerivedFrom("::proto2::MessageLite"));

    // A method's name starts with "add_" might not mean it's an add field
    // call; it could be the getter for a proto field of which the name starts
    // with "add_". So we exlude const methods.
    const auto AddFieldMethodDecl =
        cxxMethodDecl(matchesName("::add_"), unless(isConst()));
    AddMatcher(ProtoDecl, ProtoVarDeclName, ProtoVarDeclStmtName,
               AddFieldMethodDecl, ProtoAddFieldCallName, Finder);
  }
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
  const auto *ProtoVarDecl = Result.Nodes.getNodeAs<VarDecl>(ProtoVarDeclName);
  const auto *ProtoAddFieldCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>(ProtoAddFieldCallName);
  const auto *LoopEndExpr = Result.Nodes.getNodeAs<Expr>(LoopEndExprName);
  const auto *LoopParent = Result.Nodes.getNodeAs<CompoundStmt>(LoopParentName);

  const CXXMemberCallExpr *AppendCall =
      VectorAppendCall ? VectorAppendCall : ProtoAddFieldCall;
  assert(AppendCall && "no append call expression");

  const Stmt *LoopStmt = ForLoop;
  if (!LoopStmt)
    LoopStmt = RangeLoop;

  const auto *TargetVarDecl = VectorVarDecl;
  if (!TargetVarDecl)
    TargetVarDecl = ProtoVarDecl;

  llvm::SmallPtrSet<const DeclRefExpr *, 16> AllVarRefs =
      utils::decl_ref_expr::allDeclRefExprs(*TargetVarDecl, *LoopParent,
                                            *Context);
  for (const auto *Ref : AllVarRefs) {
    // Skip cases where there are usages (defined as DeclRefExpr that refers
    // to "v") of vector variable / proto variable `v` before the for loop. We
    // consider these usages are operations causing memory preallocation (e.g.
    // "v.resize(n)", "v.reserve(n)").
    //
    // FIXME: make it more intelligent to identify the pre-allocating
    // operations before the for loop.
    if (SM.isBeforeInTranslationUnit(Ref->getLocation(),
                                     LoopStmt->getBeginLoc())) {
      return;
    }
  }

  std::string PartialReserveStmt;
  if (VectorAppendCall != nullptr) {
    PartialReserveStmt = ".reserve";
  } else {
    llvm::StringRef FieldName = ProtoAddFieldCall->getMethodDecl()->getName();
    FieldName.consume_front("add_");
    std::string MutableFieldName = ("mutable_" + FieldName).str();
    PartialReserveStmt = "." + MutableFieldName +
                         "()->Reserve"; // e.g., ".mutable_xxx()->Reserve"
  }

  llvm::StringRef VarName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          AppendCall->getImplicitObjectArgument()->getSourceRange()),
      SM, Context->getLangOpts());

  std::string ReserveSize;
  // Handle for-range loop cases.
  if (RangeLoop) {
    // Get the range-expression in a for-range statement represented as
    // `for (range-declarator: range-expression)`.
    StringRef RangeInitExpName =
        Lexer::getSourceText(CharSourceRange::getTokenRange(
                                 RangeLoop->getRangeInit()->getSourceRange()),
                             SM, Context->getLangOpts());
    ReserveSize = (RangeInitExpName + ".size()").str();
  } else if (ForLoop) {
    // Handle counter-based loop cases.
    StringRef LoopEndSource = Lexer::getSourceText(
        CharSourceRange::getTokenRange(LoopEndExpr->getSourceRange()), SM,
        Context->getLangOpts());
    ReserveSize = LoopEndSource;
  }

  auto Diag = diag(AppendCall->getBeginLoc(),
                   "%0 is called inside a loop; consider pre-allocating the "
                   "container capacity before the loop")
              << AppendCall->getMethodDecl()->getDeclName();
  if (!ReserveSize.empty()) {
    std::string ReserveStmt =
        (VarName + PartialReserveStmt + "(" + ReserveSize + ");\n").str();
    Diag << FixItHint::CreateInsertion(LoopStmt->getBeginLoc(), ReserveStmt);
  }
}

} // namespace performance
} // namespace tidy
} // namespace clang
