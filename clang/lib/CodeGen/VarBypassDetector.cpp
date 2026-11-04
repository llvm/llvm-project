//===--- VarBypassDetector.cpp - Bypass jumps detector ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VarBypassDetector.h"

#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace CodeGen;

/// True if the body contains a goto, switch, or indirect goto. Lets Init()
/// skip scope building for the common jump-free function, this is a minor
/// optimization win.
static bool hasJumpStmts(const Stmt *Body) {
  llvm::SmallVector<const Stmt *, 32> Worklist;
  Worklist.push_back(Body);
  while (!Worklist.empty()) {
    const Stmt *S = Worklist.pop_back_val();
    if (!S)
      continue;
    switch (S->getStmtClass()) {
    case Stmt::GotoStmtClass:
    case Stmt::SwitchStmtClass:
    case Stmt::IndirectGotoStmtClass:
      return true;
    case Stmt::GCCAsmStmtClass:
      if (cast<GCCAsmStmt>(S)->isAsmGoto())
        return true;
      break;
    default:
      break;
    }
    for (const Stmt *Child : S->children())
      Worklist.push_back(Child);
  }
  return false;
}

/// Clear the object and pre-process for the given statement, usually function
/// body statement.
void VarBypassDetector::Init(CodeGenModule &CGM, const Stmt *Body) {
  FromScopes.clear();
  ToScopes.clear();
  Bypasses.clear();
  BypassedStmts.clear();
  BackwardBypassedVars.clear();
  BackwardBypassedCLs.clear();
  BackwardBypassedCLsInBlock.clear();
  BypassedVarsAtSource.clear();
  Scopes = {{~0U, nullptr, nullptr, nullptr, 0}};
  CurEnclosingBlock = nullptr;
  CurEnclosingBlockScope = 0;
  AlwaysBypassed = false;
  if (!hasJumpStmts(Body))
    return;
  unsigned ParentScope = 0;
  AlwaysBypassed = !BuildScopeInformation(CGM, Body, ParentScope);
  if (!AlwaysBypassed)
    Detect(CGM);
}

/// Build scope information for a declaration that is part of a DeclStmt.
/// Returns false if we failed to build scope information and can't tell for
/// which vars are being bypassed.
bool VarBypassDetector::BuildScopeInformation(CodeGenModule &CGM, const Decl *D,
                                              unsigned &ParentScope) {
  const VarDecl *VD = dyn_cast<VarDecl>(D);
  if (VD && VD->hasLocalStorage()) {
    Scopes.push_back(
        {ParentScope, VD, nullptr, CurEnclosingBlock, CurEnclosingBlockScope});
    ParentScope = Scopes.size() - 1;
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    if (const Expr *Init = VD->getInit())
      return BuildScopeInformation(CGM, Init, ParentScope);

  return true;
}

/// Walk through the statements, adding any labels or gotos to
/// LabelAndGotoScopes and recursively walking the AST as needed.
/// Returns false if we failed to build scope information and can't tell for
/// which vars are being bypassed.
bool VarBypassDetector::BuildScopeInformation(CodeGenModule &CGM, const Stmt *S,
                                              unsigned &origParentScope) {
  // If this is a statement, rather than an expression, scopes within it don't
  // propagate out into the enclosing scope. Otherwise we have to worry about
  // block literals, which have the lifetime of their enclosing statement.
  unsigned independentParentScope = origParentScope;
  unsigned &ParentScope =
      ((isa<Expr>(S) && !isa<StmtExpr>(S)) ? origParentScope
                                           : independentParentScope);

  const CompoundStmt *PrevBlock = CurEnclosingBlock;
  unsigned PrevBlockScope = CurEnclosingBlockScope;
  llvm::scope_exit RestoreBlock([&] {
    CurEnclosingBlock = PrevBlock;
    CurEnclosingBlockScope = PrevBlockScope;
  });

  unsigned StmtsToSkip = 0u;

  switch (S->getStmtClass()) {
  case Stmt::IndirectGotoStmtClass:
    return false;

  case Stmt::CompoundStmtClass: {
    const auto *CS = cast<CompoundStmt>(S);
    Scopes.push_back({ParentScope, nullptr, nullptr, CS, ParentScope});
    ParentScope = Scopes.size() - 1;
    CurEnclosingBlock = CS;
    CurEnclosingBlockScope = ParentScope;
    break;
  }

  case Stmt::CompoundLiteralExprClass: {
    const auto *CLE = cast<CompoundLiteralExpr>(S);
    if (!CGM.getLangOpts().CPlusPlus && !CLE->isFileScope()) {
      Scopes.push_back({origParentScope, nullptr, CLE, CurEnclosingBlock,
                        CurEnclosingBlockScope});
      origParentScope = Scopes.size() - 1;
    }
    break;
  }

  case Stmt::GCCAsmStmtClass:
    if (cast<GCCAsmStmt>(S)->isAsmGoto())
      FromScopes.push_back({S, ParentScope});
    break;

  case Stmt::SwitchStmtClass:
    if (const Stmt *Init = cast<SwitchStmt>(S)->getInit()) {
      if (!BuildScopeInformation(CGM, Init, ParentScope))
        return false;
      ++StmtsToSkip;
    }
    if (const VarDecl *Var = cast<SwitchStmt>(S)->getConditionVariable()) {
      if (!BuildScopeInformation(CGM, Var, ParentScope))
        return false;
      ++StmtsToSkip;
    }
    [[fallthrough]];

  case Stmt::GotoStmtClass:
    FromScopes.push_back({S, ParentScope});
    break;

  case Stmt::DeclStmtClass: {
    const DeclStmt *DS = cast<DeclStmt>(S);
    for (auto *I : DS->decls())
      if (!BuildScopeInformation(CGM, I, origParentScope))
        return false;
    return true;
  }

  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::LabelStmtClass:
    llvm_unreachable("the loop below handles labels and cases");
    break;

  default:
    break;
  }

  for (const Stmt *SubStmt : S->children()) {
    if (!SubStmt)
      continue;
    if (StmtsToSkip) {
      --StmtsToSkip;
      continue;
    }

    // Cases, labels, and defaults aren't "scope parents".  It's also
    // important to handle these iteratively instead of recursively in
    // order to avoid blowing out the stack.
    while (true) {
      const Stmt *Next;
      if (const SwitchCase *SC = dyn_cast<SwitchCase>(SubStmt))
        Next = SC->getSubStmt();
      else if (const LabelStmt *LS = dyn_cast<LabelStmt>(SubStmt))
        Next = LS->getSubStmt();
      else
        break;

      ToScopes[SubStmt] = ParentScope;
      SubStmt = Next;
    }

    // Recursively walk the AST.
    bool Result;
    CGM.runWithSufficientStackSpace(S->getEndLoc(), [&] {
      Result = BuildScopeInformation(CGM, SubStmt, ParentScope);
    });
    if (!Result)
      return false;
  }
  return true;
}

/// Checks each jump and stores each variable declaration they bypass.
void VarBypassDetector::Detect(CodeGenModule &CGM) {
  for (const auto &S : FromScopes) {
    const Stmt *St = S.first;
    unsigned from = S.second;
    if (const GotoStmt *GS = dyn_cast<GotoStmt>(St)) {
      if (const LabelStmt *LS = GS->getLabel()->getStmt())
        Detect(CGM, from, ToScopes[LS], GS);
    } else if (const SwitchStmt *SS = dyn_cast<SwitchStmt>(St)) {
      // Keyed by the switch, not by each case: the dispatch is the jump
      // source, and that is where the initialization is emitted.
      for (const SwitchCase *SC = SS->getSwitchCaseList(); SC;
           SC = SC->getNextSwitchCase()) {
        Detect(CGM, from, ToScopes[SC], SS);
      }
    } else if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(St)) {
      for (const AddrLabelExpr *L : GAS->labels())
        if (const LabelStmt *LS = L->getLabel()->getStmt())
          Detect(CGM, from, ToScopes[LS], GAS);
    } else {
      llvm_unreachable("goto, switch, or asm goto was expected");
    }
  }
}

/// Checks the jump and stores each variable declaration or compound literal it
/// bypasses.
void VarBypassDetector::Detect(CodeGenModule &CGM, unsigned From, unsigned To,
                               const Stmt *Source) {
  while (From != To) {
    if (From < To) {
      assert(Scopes[To].ParentScope < To);
      const ScopeEntry &ScopeTo = Scopes[To];
      To = ScopeTo.ParentScope;
      if (ScopeTo.VD) {
        Bypasses.insert(ScopeTo.VD);
        BypassedVarsAtSource[Source].insert(ScopeTo.VD);
      }
      if (ScopeTo.CLE)
        BypassedStmts.insert(ScopeTo.CLE);
      if (ScopeTo.Block)
        BypassedStmts.insert(ScopeTo.Block);
    } else {
      assert(Scopes[From].ParentScope < From);
      const ScopeEntry &ScopeFrom = Scopes[From];
      if (!CGM.getLangOpts().CPlusPlus && ScopeFrom.EnclosingBlockScope <= To) {
        if (ScopeFrom.VD)
          BackwardBypassedVars.insert(ScopeFrom.VD);
        if (ScopeFrom.CLE && ScopeFrom.Block &&
            !ScopeFrom.CLE->getType()->isVariablyModifiedType()) {
          BackwardBypassedCLs.insert(ScopeFrom.CLE);
          BackwardBypassedCLsInBlock[ScopeFrom.Block].insert(ScopeFrom.CLE);
        }
      }
      From = ScopeFrom.ParentScope;
    }
  }
}
