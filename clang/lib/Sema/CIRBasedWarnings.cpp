//=- CIRBasedWarnings.cpp - Sema warnings based on libAnalysis -*- C++ ----*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines analysis_warnings::[Policy,Executor].
// Together they are used by Sema to issue warnings based on inexpensive
// static analysis algorithms using ClangIR.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/CIRBasedWarnings.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/CIR/IR/CIRDialect.h"

#include <algorithm>
#include <deque>
#include <iterator>

using namespace clang;

namespace {
///
/// Helpers
///
class reverse_children {
  llvm::SmallVector<Stmt *, 12> childrenBuf;
  ArrayRef<Stmt *> children;

public:
  reverse_children(Stmt *S);

  using iterator = ArrayRef<Stmt *>::reverse_iterator;

  iterator begin() const { return children.rbegin(); }
  iterator end() const { return children.rend(); }
};

// FIXME: we might not even need this.
reverse_children::reverse_children(Stmt *S) {
  if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
    children = CE->getRawSubExprs();
    return;
  }
  switch (S->getStmtClass()) {
  // Note: Fill in this switch with more cases we want to optimize.
  case Stmt::InitListExprClass: {
    InitListExpr *IE = cast<InitListExpr>(S);
    children = llvm::ArrayRef(reinterpret_cast<Stmt **>(IE->getInits()),
                              IE->getNumInits());
    return;
  }
  default:
    break;
  }

  // Default case for all other statements.
  for (Stmt *SubStmt : S->children())
    childrenBuf.push_back(SubStmt);

  // This needs to be done *after* childrenBuf has been populated.
  children = childrenBuf;
}

///
/// CIRBuilder
///

/// CIRBuilder - This class implements CIR construction from an AST.
class CIRBuilder {
public:
  typedef int CIRUnit;
  explicit CIRBuilder(ASTContext *astContext) : Context(astContext) {}

  ASTContext *Context;

  // buildCFG - Used by external clients to construct the CFG.
  // std::unique_ptr<CFG> buildCIR(const Decl *D, Stmt *Statement);
  void buildCIR(const Decl *D, Stmt *Statement);

private:
  // Visitors to walk an AST and construct CIR.
  CIRUnit *VisitImplicitCastExpr(ImplicitCastExpr *E);
  CIRUnit *VisitCompoundStmt(CompoundStmt *C);
  CIRUnit *VisitDeclStmt(DeclStmt *DS);

  // Basic components
  CIRUnit *Visit(Stmt *S);
  CIRUnit *VisitStmt(Stmt *S);
  CIRUnit *VisitChildren(Stmt *S);
};

using CIRUnit = CIRBuilder::CIRUnit;

///
/// Basic visitors
///

/// Visit - Walk the subtree of a statement and add extra
///   blocks for ternary operators, &&, and ||.  We also process "," and
///   DeclStmts (which may contain nested control-flow).
CIRUnit *CIRBuilder::Visit(Stmt *S) {
  if (!S) {
    return nullptr;
  }

  // if (Expr *E = dyn_cast<Expr>(S))
  //  S = E->IgnoreParens();

  switch (S->getStmtClass()) {
  default:
    return VisitStmt(S);

  case Stmt::CompoundStmtClass:
    return VisitCompoundStmt(cast<CompoundStmt>(S));

  case Stmt::ImplicitCastExprClass:
    return VisitImplicitCastExpr(cast<ImplicitCastExpr>(S));

  case Stmt::DeclStmtClass:
    return VisitDeclStmt(cast<DeclStmt>(S));
  }
}

CIRUnit *CIRBuilder::VisitStmt(Stmt *S) {
  // FIXME: do work.
  return VisitChildren(S);
}

/// VisitChildren - Visit the children of a Stmt.
CIRUnit *CIRBuilder::VisitChildren(Stmt *S) {
  // Visit the children in their reverse order so that they appear in
  // left-to-right (natural) order in the CFG.
  // reverse_children RChildren(S);
  // for (Stmt *Child : RChildren) {
  //  if (Child)
  //    if (CIRUnit *R = Visit(Child))
  //      B = R;
  // }
  return nullptr; // B;
}

///
/// Other visitors
///
CIRUnit *CIRBuilder::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  // FIXME: do work.
  return nullptr;
}

CIRUnit *CIRBuilder::VisitCompoundStmt(CompoundStmt *C) {
  // FIXME: do work.
  return nullptr;
}

CIRUnit *CIRBuilder::VisitDeclStmt(DeclStmt *DS) {
  // FIXME: do work.
  return nullptr;
}

} // namespace

///
/// CIRBasedWarnings
///
static unsigned isEnabled(DiagnosticsEngine &D, unsigned diag) {
  return (unsigned)!D.isIgnored(diag, SourceLocation());
}

sema::CIRBasedWarnings::CIRBasedWarnings(Sema &s) : S(s) {

  using namespace diag;
  DiagnosticsEngine &D = S.getDiagnostics();

  DefaultPolicy.enableCheckUnreachable =
      isEnabled(D, warn_unreachable) || isEnabled(D, warn_unreachable_break) ||
      isEnabled(D, warn_unreachable_return) ||
      isEnabled(D, warn_unreachable_loop_increment);

  DefaultPolicy.enableThreadSafetyAnalysis = isEnabled(D, warn_double_lock);

  DefaultPolicy.enableConsumedAnalysis =
      isEnabled(D, warn_use_in_invalid_state);
}

// We need this here for unique_ptr with forward declared class.
sema::CIRBasedWarnings::~CIRBasedWarnings() = default;

static void flushDiagnostics(Sema &S, const sema::FunctionScopeInfo *fscope) {
  for (const auto &D : fscope->PossiblyUnreachableDiags)
    S.Diag(D.Loc, D.PD);
}

void clang::sema::CIRBasedWarnings::IssueWarnings(
    sema::AnalysisBasedWarnings::Policy P, sema::FunctionScopeInfo *fscope,
    const Decl *D, QualType BlockType) {
  // We avoid doing analysis-based warnings when there are errors for
  // two reasons:
  // (1) The CFGs often can't be constructed (if the body is invalid), so
  //     don't bother trying.
  // (2) The code already has problems; running the analysis just takes more
  //     time.
  DiagnosticsEngine &Diags = S.getDiagnostics();

  // Do not do any analysis if we are going to just ignore them.
  if (Diags.getIgnoreAllWarnings() ||
      (Diags.getSuppressSystemWarnings() &&
       S.SourceMgr.isInSystemHeader(D->getLocation())))
    return;

  // For code in dependent contexts, we'll do this at instantiation time.
  if (cast<DeclContext>(D)->isDependentContext())
    return;

  if (S.hasUncompilableErrorOccurred()) {
    // Flush out any possibly unreachable diagnostics.
    flushDiagnostics(S, fscope);
    return;
  }

  const Stmt *Body = D->getBody();
  assert(Body);

  // TODO: up to this point this behaves the same as
  // AnalysisBasedWarnings::IssueWarnings

  // Unlike Clang CFG, we share CIR state between each analyzed function,
  // retrieve or create a new context.
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::cir::CIRDialect>();
}

void clang::sema::CIRBasedWarnings::PrintStats() const {
  llvm::errs() << "\n*** CIR Based Warnings Stats:\n";
}
