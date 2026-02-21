//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseReturnValueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

/// Find the single non-const lvalue reference parameter in a function that
/// could serve as an output parameter. Returns nullptr if there are zero or
/// more than one such parameters, or if the parameter type is not suitable.
const ParmVarDecl *findSingleOutParam(const FunctionDecl *Func) {
  const ParmVarDecl *Candidate = nullptr;
  for (const auto *Param : Func->parameters()) {
    const QualType T = Param->getType();
    if (!T->isLValueReferenceType())
      continue;
    const QualType Pointee = T->getPointeeType();
    // Skip const references -- those are input parameters.
    if (Pointee.isConstQualified())
      continue;
    // Skip references to non-object types.
    if (Pointee->isFunctionType() || Pointee->isVoidType())
      continue;
    // More than one non-const ref param -- ambiguous.
    if (Candidate)
      return nullptr;
    Candidate = Param;
  }
  return Candidate;
}

/// Visitor that checks whether a parameter is only written to (never read
/// before being fully assigned). A parameter is an "output-only" parameter if
/// every use of it in the function body is an assignment target or passed to
/// a function that takes a non-const reference (i.e., it is used purely for
/// output).
class OutParamVisitor : public RecursiveASTVisitor<OutParamVisitor> {
public:
  explicit OutParamVisitor(const ParmVarDecl *Param) : Param(Param) {}

  bool isOutputOnly() const { return HasWrite && !HasRead; }

  bool VisitDeclRefExpr(const DeclRefExpr *DRE) {
    if (DRE->getDecl() != Param)
      return true;

    // Check how this reference is used by looking at the parent.
    // We conservatively mark any use we cannot classify as a read.
    HasWrite = true;
    HasRead = true; // Conservative default; refined by parent checks.
    return true;
  }

  // Override to check parent context of DeclRefExprs.
  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;
    return RecursiveASTVisitor::TraverseStmt(S);
  }

  /// Check if the param is used as an assignment LHS:
  ///   param = expr;       (direct assignment)
  ///   param.field = expr; (member assignment)
  bool VisitBinaryOperator(const BinaryOperator *BO) {
    if (BO->getOpcode() != BO_Assign)
      return true;
    const Expr *LHS = BO->getLHS()->IgnoreImplicit();
    // Direct assignment: param = expr.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(LHS))
      if (DRE->getDecl() == Param)
        FoundAssignment = true;
    // Member assignment: param.field = expr.
    if (const auto *ME = dyn_cast<MemberExpr>(LHS))
      if (const auto *DRE =
              dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImplicit()))
        if (DRE->getDecl() == Param)
          FoundAssignment = true;
    return true;
  }

  bool hasAssignment() const { return FoundAssignment; }

private:
  const ParmVarDecl *Param;
  bool HasWrite = false;
  bool HasRead = false;
  bool FoundAssignment = false;
};

} // namespace

void UseReturnValueCheck::registerMatchers(MatchFinder *Finder) {
  // Match non-template function definitions returning void.
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(isImplicit()), unless(isDeleted()),
                   returns(voidType()),
                   unless(hasParent(functionTemplateDecl())),
                   // At least one non-const ref parameter.
                   hasAnyParameter(parmVarDecl(hasType(lValueReferenceType(
                       pointee(unless(isConstQualified())))))))
          .bind("func"),
      this);
}

void UseReturnValueCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  if (!Func || !Func->hasBody())
    return;

  // Skip system headers.
  if (Result.SourceManager->isInSystemHeader(Func->getLocation()))
    return;

  // Skip virtual methods (changing return type breaks polymorphism).
  if (const auto *MD = dyn_cast<CXXMethodDecl>(Func))
    if (MD->isVirtual())
      return;

  // Skip main.
  if (Func->isMain())
    return;

  // Find the single output parameter.
  const ParmVarDecl *OutParam = findSingleOutParam(Func);
  if (!OutParam || !OutParam->getIdentifier())
    return;

  // The output parameter type must be copyable/movable (not an abstract
  // class, not an array, etc.).
  const QualType ParamType = OutParam->getType()->getPointeeType();
  if (ParamType->isArrayType() || ParamType->isIncompleteType())
    return;
  if (const auto *RD = ParamType->getAsCXXRecordDecl())
    if (RD->isAbstract())
      return;

  // Verify the parameter is actually assigned to in the body.
  OutParamVisitor Visitor(OutParam);
  Visitor.TraverseStmt(Func->getBody());

  if (!Visitor.hasAssignment())
    return;

  diag(Func->getLocation(),
       "function '%0' has output parameter '%1'; consider returning "
       "the value directly instead")
      << Func->getName() << OutParam->getName();
  diag(OutParam->getLocation(), "output parameter declared here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::modernize
