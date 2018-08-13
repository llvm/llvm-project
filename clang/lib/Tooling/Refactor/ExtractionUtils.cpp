//===--- ExtractionUtils.cpp - Extraction helper functions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExtractionUtils.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

Optional<StringRef> tooling::extract::nameForExtractedVariable(const Expr *E) {
  if (const auto *Call = dyn_cast<CallExpr>(E)) {
    if (const auto *Fn = Call->getDirectCallee())
      return Fn->getName();
  } else if (const auto *Msg = dyn_cast<ObjCMessageExpr>(E)) {
    if (const auto *M = Msg->getMethodDecl()) {
      if (M->getSelector().isUnarySelector())
        return M->getSelector().getNameForSlot(0);
    }
  } else if (const auto *PRE = dyn_cast<ObjCPropertyRefExpr>(E)) {
    if (PRE->isImplicitProperty()) {
      if (const auto *M = PRE->getImplicitPropertyGetter())
        return M->getSelector().getNameForSlot(0);
    } else if (const auto *Prop = PRE->getExplicitProperty())
      return Prop->getName();
  }
  return None;
}

namespace {

/// Checks if a set of expressions is directly contained in some AST region.
class StmtReachabilityChecker
    : public RecursiveASTVisitor<StmtReachabilityChecker> {
  const llvm::SmallPtrSetImpl<const Stmt *> &Expressions;
  unsigned Count = 0;

  StmtReachabilityChecker(
      const llvm::SmallPtrSetImpl<const Stmt *> &Expressions)
      : Expressions(Expressions) {}

  bool areAllExpressionsReached() const { return Count == Expressions.size(); }

public:
  bool VisitStmt(const Stmt *S) {
    if (Expressions.count(S)) {
      ++Count;
      if (areAllExpressionsReached())
        return false;
    }
    return true;
  }

  static bool areAllExpressionsReachableFrom(
      CompoundStmt *S, const llvm::SmallPtrSetImpl<const Stmt *> &Expressions) {
    StmtReachabilityChecker Checker(Expressions);
    Checker.TraverseStmt(S);
    return Checker.areAllExpressionsReached();
  }
};

/// Figures out where the extracted variable should go.
class ExtractedVariableInsertionLocFinder
    : public RecursiveASTVisitor<ExtractedVariableInsertionLocFinder> {
  llvm::SmallPtrSet<const Stmt *, 4> Expressions;
  llvm::SmallVector<std::pair<CompoundStmt *, const Stmt *>, 4>
      InsertionCandidateStack;
  bool IsPrevCompoundStmt = false;

public:
  SourceLocation Loc;

  /// Initializes the insertion location finder using the set of duplicate
  /// \p Expressions from one function.
  ExtractedVariableInsertionLocFinder(ArrayRef<const Expr *> Expressions) {
    for (const Expr *E : Expressions)
      this->Expressions.insert(E);
  }

  bool TraverseStmt(Stmt *S) {
    if (!S)
      return RecursiveASTVisitor::TraverseStmt(S);
    if (IsPrevCompoundStmt && !InsertionCandidateStack.empty())
      InsertionCandidateStack.back().second = S;
    llvm::SaveAndRestore<bool> IsPrevCompoundStmtTracker(IsPrevCompoundStmt,
                                                         false);
    if (auto *CS = dyn_cast<CompoundStmt>(S)) {
      IsPrevCompoundStmt = true;
      InsertionCandidateStack.emplace_back(CS, nullptr);
      RecursiveASTVisitor::TraverseStmt(S);
      InsertionCandidateStack.pop_back();
      return true;
    }
    return RecursiveASTVisitor::TraverseStmt(S);
  }

  bool VisitStmt(const Stmt *S) {
    if (Expressions.count(S)) {
      // The insertion location should be in the first compound statement that
      // includes all of the expressions as descendants as we want the new
      // variable to be visible to all uses.
      for (auto I = InsertionCandidateStack.rbegin(),
                E = InsertionCandidateStack.rend();
           I != E; ++I) {
        if (StmtReachabilityChecker::areAllExpressionsReachableFrom(
                I->first, Expressions) &&
            I->second) {
          Loc = I->second->getBeginLoc();
          break;
        }
      }
      return false;
    }
    return true;
  }
};

} // end anonymous namespace

SourceLocation tooling::extract::locationForExtractedVariableDeclaration(
    ArrayRef<const Expr *> Expressions, const Decl *ParentDecl,
    const SourceManager &SM) {
  ExtractedVariableInsertionLocFinder LocFinder(Expressions);
  LocFinder.TraverseDecl(const_cast<Decl *>(ParentDecl));
  SourceLocation Result = LocFinder.Loc;
  if (Result.isValid() && Result.isMacroID())
    return SM.getExpansionLoc(Result);
  return Result;
}
