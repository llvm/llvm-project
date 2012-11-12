//==- CheckSizeofPointer.cpp - Check for sizeof on pointers ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a check for unintended use of sizeof() on pointer
//  expressions.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"

using namespace clang;
using namespace ento;

namespace {
class WalkAST : public StmtVisitor<WalkAST> {
  BugReporter &BR;
  AnalysisDeclContext* AC;

public:
  WalkAST(BugReporter &br, AnalysisDeclContext* ac) : BR(br), AC(ac) {}
  void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
  void VisitStmt(Stmt *S) { VisitChildren(S); }
  void VisitChildren(Stmt *S);
};
}

void WalkAST::VisitChildren(Stmt *S) {
  for (Stmt::child_iterator I = S->child_begin(), E = S->child_end(); I!=E; ++I)
    if (Stmt *child = *I)
      Visit(child);
}

// CWE-467: Use of sizeof() on a Pointer Type
void WalkAST::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E) {
  if (E->getKind() != UETT_SizeOf)
    return;

  // If an explicit type is used in the code, usually the coder knows what he is
  // doing.
  if (E->isArgumentType())
    return;

  QualType T = E->getTypeOfArgument();
  if (T->isPointerType()) {

    // Many false positives have the form 'sizeof *p'. This is reasonable 
    // because people know what they are doing when they intentionally 
    // dereference the pointer.
    Expr *ArgEx = E->getArgumentExpr();
    if (!isa<DeclRefExpr>(ArgEx->IgnoreParens()))
      return;

    SourceRange R = ArgEx->getSourceRange();
    PathDiagnosticLocation ELoc =
      PathDiagnosticLocation::createBegin(E, BR.getSourceManager(), AC);
    BR.EmitBasicReport(AC->getDecl(),
                       "Potential unintended use of sizeof() on pointer type",
                       "Logic",
                       "The code calls sizeof() on a pointer type. "
                       "This can produce an unexpected result.",
                       ELoc, &R, 1);
  }
}

//===----------------------------------------------------------------------===//
// SizeofPointerChecker
//===----------------------------------------------------------------------===//

namespace {
class SizeofPointerChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    WalkAST walker(BR, mgr.getAnalysisDeclContext(D));
    walker.Visit(D->getBody());
  }
};
}

void ento::registerSizeofPointerChecker(CheckerManager &mgr) {
  mgr.registerChecker<SizeofPointerChecker>();
}
