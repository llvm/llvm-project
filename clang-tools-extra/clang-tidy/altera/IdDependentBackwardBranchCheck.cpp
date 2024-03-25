//===--- IdDependentBackwardBranchCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdDependentBackwardBranchCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::altera {

void IdDependentBackwardBranchCheck::registerMatchers(MatchFinder *Finder) {
  // Prototype to identify all variables which hold a thread-variant ID.
  // First Matcher just finds all the direct assignments of either ID call.
  const auto ThreadID = expr(hasDescendant(callExpr(callee(functionDecl(
      anyOf(hasName("get_global_id"), hasName("get_local_id")))))));

  const auto RefVarOrField = forEachDescendant(
      stmt(anyOf(declRefExpr(to(varDecl())).bind("assign_ref_var"),
                 memberExpr(member(fieldDecl())).bind("assign_ref_field"))));

  Finder->addMatcher(
      compoundStmt(
          // Bind on actual get_local/global_id calls.
          forEachDescendant(
              stmt(
                  anyOf(declStmt(hasDescendant(varDecl(hasInitializer(ThreadID))
                                                   .bind("tid_dep_var"))),
                        binaryOperator(
                            isAssignmentOperator(), hasRHS(ThreadID),
                            hasLHS(anyOf(
                                declRefExpr(to(varDecl().bind("tid_dep_var"))),
                                memberExpr(member(
                                    fieldDecl().bind("tid_dep_field"))))))))
                  .bind("straight_assignment"))),
      this);

  // Bind all VarDecls that include an initializer with a variable DeclRefExpr
  // (in case it is ID-dependent).
  Finder->addMatcher(
      stmt(forEachDescendant(
          varDecl(hasInitializer(RefVarOrField)).bind("pot_tid_var"))),
      this);

  // Bind all VarDecls that are assigned a value with a variable DeclRefExpr (in
  // case it is ID-dependent).
  Finder->addMatcher(
      stmt(forEachDescendant(binaryOperator(
          allOf(isAssignmentOperator(), hasRHS(RefVarOrField),
                hasLHS(anyOf(
                    declRefExpr(to(varDecl().bind("pot_tid_var"))),
                    memberExpr(member(fieldDecl().bind("pot_tid_field"))))))))),
      this);

  // Second Matcher looks for branch statements inside of loops and bind on the
  // condition expression IF it either calls an ID function or has a variable
  // DeclRefExpr. DeclRefExprs are checked later to confirm whether the variable
  // is ID-dependent.
  const auto CondExpr =
      expr(anyOf(hasDescendant(callExpr(callee(functionDecl(
                                            anyOf(hasName("get_global_id"),
                                                  hasName("get_local_id")))))
                                   .bind("id_call")),
                 hasDescendant(stmt(anyOf(declRefExpr(to(varDecl())),
                                          memberExpr(member(fieldDecl())))))))
          .bind("cond_expr");
  Finder->addMatcher(stmt(anyOf(forStmt(hasCondition(CondExpr)),
                                doStmt(hasCondition(CondExpr)),
                                whileStmt(hasCondition(CondExpr))))
                         .bind("backward_branch"),
                     this);
}

IdDependentBackwardBranchCheck::IdDependencyRecord *
IdDependentBackwardBranchCheck::hasIdDepVar(const Expr *Expression) {
  if (const auto *Declaration = dyn_cast<DeclRefExpr>(Expression)) {
    // It is a DeclRefExpr, so check if it's an ID-dependent variable.
    const auto *CheckVariable = dyn_cast<VarDecl>(Declaration->getDecl());
    auto FoundVariable = IdDepVarsMap.find(CheckVariable);
    if (FoundVariable == IdDepVarsMap.end())
      return nullptr;
    return &(FoundVariable->second);
  }
  for (const auto *Child : Expression->children())
    if (const auto *ChildExpression = dyn_cast<Expr>(Child))
      if (IdDependencyRecord *Result = hasIdDepVar(ChildExpression))
        return Result;
  return nullptr;
}

IdDependentBackwardBranchCheck::IdDependencyRecord *
IdDependentBackwardBranchCheck::hasIdDepField(const Expr *Expression) {
  if (const auto *MemberExpression = dyn_cast<MemberExpr>(Expression)) {
    const auto *CheckField =
        dyn_cast<FieldDecl>(MemberExpression->getMemberDecl());
    auto FoundField = IdDepFieldsMap.find(CheckField);
    if (FoundField == IdDepFieldsMap.end())
      return nullptr;
    return &(FoundField->second);
  }
  for (const auto *Child : Expression->children())
    if (const auto *ChildExpression = dyn_cast<Expr>(Child))
      if (IdDependencyRecord *Result = hasIdDepField(ChildExpression))
        return Result;
  return nullptr;
}

void IdDependentBackwardBranchCheck::saveIdDepVar(const Stmt *Statement,
                                                  const VarDecl *Variable) {
  // Record that this variable is thread-dependent.
  IdDepVarsMap[Variable] =
      IdDependencyRecord(Variable, Variable->getBeginLoc(),
                         Twine("assignment of ID-dependent variable ") +
                             Variable->getNameAsString());
}

void IdDependentBackwardBranchCheck::saveIdDepField(const Stmt *Statement,
                                                    const FieldDecl *Field) {
  // Record that this field is thread-dependent.
  IdDepFieldsMap[Field] = IdDependencyRecord(
      Field, Statement->getBeginLoc(),
      Twine("assignment of ID-dependent field ") + Field->getNameAsString());
}

void IdDependentBackwardBranchCheck::saveIdDepVarFromReference(
    const DeclRefExpr *RefExpr, const MemberExpr *MemExpr,
    const VarDecl *PotentialVar) {
  // If the variable is already in IdDepVarsMap, ignore it.
  if (IdDepVarsMap.find(PotentialVar) != IdDepVarsMap.end())
    return;
  std::string Message;
  llvm::raw_string_ostream StringStream(Message);
  StringStream << "inferred assignment of ID-dependent value from "
                  "ID-dependent ";
  if (RefExpr) {
    const auto *RefVar = dyn_cast<VarDecl>(RefExpr->getDecl());
    // If variable isn't ID-dependent, but RefVar is.
    if (IdDepVarsMap.find(RefVar) != IdDepVarsMap.end())
      StringStream << "variable " << RefVar->getNameAsString();
  }
  if (MemExpr) {
    const auto *RefField = dyn_cast<FieldDecl>(MemExpr->getMemberDecl());
    // If variable isn't ID-dependent, but RefField is.
    if (IdDepFieldsMap.find(RefField) != IdDepFieldsMap.end())
      StringStream << "member " << RefField->getNameAsString();
  }
  IdDepVarsMap[PotentialVar] =
      IdDependencyRecord(PotentialVar, PotentialVar->getBeginLoc(), Message);
}

void IdDependentBackwardBranchCheck::saveIdDepFieldFromReference(
    const DeclRefExpr *RefExpr, const MemberExpr *MemExpr,
    const FieldDecl *PotentialField) {
  // If the field is already in IdDepFieldsMap, ignore it.
  if (IdDepFieldsMap.find(PotentialField) != IdDepFieldsMap.end())
    return;
  std::string Message;
  llvm::raw_string_ostream StringStream(Message);
  StringStream << "inferred assignment of ID-dependent member from "
                  "ID-dependent ";
  if (RefExpr) {
    const auto *RefVar = dyn_cast<VarDecl>(RefExpr->getDecl());
    // If field isn't ID-dependent, but RefVar is.
    if (IdDepVarsMap.find(RefVar) != IdDepVarsMap.end())
      StringStream << "variable " << RefVar->getNameAsString();
  }
  if (MemExpr) {
    const auto *RefField = dyn_cast<FieldDecl>(MemExpr->getMemberDecl());
    if (IdDepFieldsMap.find(RefField) != IdDepFieldsMap.end())
      StringStream << "member " << RefField->getNameAsString();
  }
  IdDepFieldsMap[PotentialField] = IdDependencyRecord(
      PotentialField, PotentialField->getBeginLoc(), Message);
}

IdDependentBackwardBranchCheck::LoopType
IdDependentBackwardBranchCheck::getLoopType(const Stmt *Loop) {
  switch (Loop->getStmtClass()) {
  case Stmt::DoStmtClass:
    return DoLoop;
  case Stmt::WhileStmtClass:
    return WhileLoop;
  case Stmt::ForStmtClass:
    return ForLoop;
  default:
    return UnknownLoop;
  }
}

void IdDependentBackwardBranchCheck::check(
    const MatchFinder::MatchResult &Result) {
  // The first half of the callback only deals with identifying and storing
  // ID-dependency information into the IdDepVars and IdDepFields maps.
  const auto *Variable = Result.Nodes.getNodeAs<VarDecl>("tid_dep_var");
  const auto *Field = Result.Nodes.getNodeAs<FieldDecl>("tid_dep_field");
  const auto *Statement = Result.Nodes.getNodeAs<Stmt>("straight_assignment");
  const auto *RefExpr = Result.Nodes.getNodeAs<DeclRefExpr>("assign_ref_var");
  const auto *MemExpr = Result.Nodes.getNodeAs<MemberExpr>("assign_ref_field");
  const auto *PotentialVar = Result.Nodes.getNodeAs<VarDecl>("pot_tid_var");
  const auto *PotentialField =
      Result.Nodes.getNodeAs<FieldDecl>("pot_tid_field");

  // Save variables and fields assigned directly through ID function calls.
  if (Statement && (Variable || Field)) {
    if (Variable)
      saveIdDepVar(Statement, Variable);
    else if (Field)
      saveIdDepField(Statement, Field);
  }

  // Save variables assigned to values of Id-dependent variables and fields.
  if ((RefExpr || MemExpr) && PotentialVar)
    saveIdDepVarFromReference(RefExpr, MemExpr, PotentialVar);

  // Save fields assigned to values of ID-dependent variables and fields.
  if ((RefExpr || MemExpr) && PotentialField)
    saveIdDepFieldFromReference(RefExpr, MemExpr, PotentialField);

  // The second part of the callback deals with checking if a branch inside a
  // loop is thread dependent.
  const auto *CondExpr = Result.Nodes.getNodeAs<Expr>("cond_expr");
  const auto *IDCall = Result.Nodes.getNodeAs<CallExpr>("id_call");
  const auto *Loop = Result.Nodes.getNodeAs<Stmt>("backward_branch");
  if (!Loop)
    return;
  LoopType Type = getLoopType(Loop);
  if (CondExpr) {
    if (IDCall) { // Conditional expression calls an ID function directly.
      diag(CondExpr->getBeginLoc(),
           "backward branch (%select{do|while|for}0 loop) is ID-dependent due "
           "to ID function call and may cause performance degradation")
          << Type;
      return;
    }
    // Conditional expression has DeclRefExpr(s), check ID-dependency.
    IdDependencyRecord *IdDepVar = hasIdDepVar(CondExpr);
    IdDependencyRecord *IdDepField = hasIdDepField(CondExpr);
    if (IdDepVar) {
      diag(CondExpr->getBeginLoc(),
           "backward branch (%select{do|while|for}0 loop) is ID-dependent due "
           "to variable reference to %1 and may cause performance degradation")
          << Type << IdDepVar->VariableDeclaration;
      diag(IdDepVar->Location, IdDepVar->Message, DiagnosticIDs::Note);
    } else if (IdDepField) {
      diag(CondExpr->getBeginLoc(),
           "backward branch (%select{do|while|for}0 loop) is ID-dependent due "
           "to member reference to %1 and may cause performance degradation")
          << Type << IdDepField->FieldDeclaration;
      diag(IdDepField->Location, IdDepField->Message, DiagnosticIDs::Note);
    }
  }
}

} // namespace clang::tidy::altera
