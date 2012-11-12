//===--- JumpDiagnostics.cpp - Protected scope jump analysis ------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JumpScopeChecker class, which is used to diagnose
// jumps that enter a protected scope in an invalid way.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/BitVector.h"
using namespace clang;

namespace {

/// JumpScopeChecker - This object is used by Sema to diagnose invalid jumps
/// into VLA and other protected scopes.  For example, this rejects:
///    goto L;
///    int a[n];
///  L:
///
class JumpScopeChecker {
  Sema &S;

  /// GotoScope - This is a record that we use to keep track of all of the
  /// scopes that are introduced by VLAs and other things that scope jumps like
  /// gotos.  This scope tree has nothing to do with the source scope tree,
  /// because you can have multiple VLA scopes per compound statement, and most
  /// compound statements don't introduce any scopes.
  struct GotoScope {
    /// ParentScope - The index in ScopeMap of the parent scope.  This is 0 for
    /// the parent scope is the function body.
    unsigned ParentScope;

    /// InDiag - The note to emit if there is a jump into this scope.
    unsigned InDiag;

    /// OutDiag - The note to emit if there is an indirect jump out
    /// of this scope.  Direct jumps always clean up their current scope
    /// in an orderly way.
    unsigned OutDiag;

    /// Loc - Location to emit the diagnostic.
    SourceLocation Loc;

    GotoScope(unsigned parentScope, unsigned InDiag, unsigned OutDiag,
              SourceLocation L)
      : ParentScope(parentScope), InDiag(InDiag), OutDiag(OutDiag), Loc(L) {}
  };

  SmallVector<GotoScope, 48> Scopes;
  llvm::DenseMap<Stmt*, unsigned> LabelAndGotoScopes;
  SmallVector<Stmt*, 16> Jumps;

  SmallVector<IndirectGotoStmt*, 4> IndirectJumps;
  SmallVector<LabelDecl*, 4> IndirectJumpTargets;
public:
  JumpScopeChecker(Stmt *Body, Sema &S);
private:
  void BuildScopeInformation(Decl *D, unsigned &ParentScope);
  void BuildScopeInformation(VarDecl *D, const BlockDecl *BDecl, 
                             unsigned &ParentScope);
  void BuildScopeInformation(Stmt *S, unsigned &origParentScope);
  
  void VerifyJumps();
  void VerifyIndirectJumps();
  void NoteJumpIntoScopes(ArrayRef<unsigned> ToScopes);
  void DiagnoseIndirectJump(IndirectGotoStmt *IG, unsigned IGScope,
                            LabelDecl *Target, unsigned TargetScope);
  void CheckJump(Stmt *From, Stmt *To, SourceLocation DiagLoc,
                 unsigned JumpDiag, unsigned JumpDiagWarning,
                 unsigned JumpDiagCXX98Compat);

  unsigned GetDeepestCommonScope(unsigned A, unsigned B);
};
} // end anonymous namespace


JumpScopeChecker::JumpScopeChecker(Stmt *Body, Sema &s) : S(s) {
  // Add a scope entry for function scope.
  Scopes.push_back(GotoScope(~0U, ~0U, ~0U, SourceLocation()));

  // Build information for the top level compound statement, so that we have a
  // defined scope record for every "goto" and label.
  unsigned BodyParentScope = 0;
  BuildScopeInformation(Body, BodyParentScope);

  // Check that all jumps we saw are kosher.
  VerifyJumps();
  VerifyIndirectJumps();
}

/// GetDeepestCommonScope - Finds the innermost scope enclosing the
/// two scopes.
unsigned JumpScopeChecker::GetDeepestCommonScope(unsigned A, unsigned B) {
  while (A != B) {
    // Inner scopes are created after outer scopes and therefore have
    // higher indices.
    if (A < B) {
      assert(Scopes[B].ParentScope < B);
      B = Scopes[B].ParentScope;
    } else {
      assert(Scopes[A].ParentScope < A);
      A = Scopes[A].ParentScope;
    }
  }
  return A;
}

typedef std::pair<unsigned,unsigned> ScopePair;

/// GetDiagForGotoScopeDecl - If this decl induces a new goto scope, return a
/// diagnostic that should be emitted if control goes over it. If not, return 0.
static ScopePair GetDiagForGotoScopeDecl(ASTContext &Context, const Decl *D) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    unsigned InDiag = 0;
    if (VD->getType()->isVariablyModifiedType())
      InDiag = diag::note_protected_by_vla;

    if (VD->hasAttr<BlocksAttr>())
      return ScopePair(diag::note_protected_by___block,
                       diag::note_exits___block);

    if (VD->hasAttr<CleanupAttr>())
      return ScopePair(diag::note_protected_by_cleanup,
                       diag::note_exits_cleanup);

    if (Context.getLangOpts().ObjCAutoRefCount && VD->hasLocalStorage()) {
      switch (VD->getType().getObjCLifetime()) {
      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        break;

      case Qualifiers::OCL_Strong:
      case Qualifiers::OCL_Weak:
        return ScopePair(diag::note_protected_by_objc_ownership,
                         diag::note_exits_objc_ownership);
      }
    }

    if (Context.getLangOpts().CPlusPlus && VD->hasLocalStorage()) {
      // C++11 [stmt.dcl]p3:
      //   A program that jumps from a point where a variable with automatic
      //   storage duration is not in scope to a point where it is in scope
      //   is ill-formed unless the variable has scalar type, class type with
      //   a trivial default constructor and a trivial destructor, a 
      //   cv-qualified version of one of these types, or an array of one of
      //   the preceding types and is declared without an initializer.

      // C++03 [stmt.dcl.p3:
      //   A program that jumps from a point where a local variable
      //   with automatic storage duration is not in scope to a point
      //   where it is in scope is ill-formed unless the variable has
      //   POD type and is declared without an initializer.

      const Expr *Init = VD->getInit();
      if (!Init)
        return ScopePair(InDiag, 0);

      const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(Init);
      if (EWC)
        Init = EWC->getSubExpr();

      const MaterializeTemporaryExpr *M = NULL;
      Init = Init->findMaterializedTemporary(M);

      SmallVector<SubobjectAdjustment, 2> Adjustments;
      Init = Init->skipRValueSubobjectAdjustments(Adjustments);

      QualType QT = Init->getType();
      if (QT.isNull())
        return ScopePair(diag::note_protected_by_variable_init, 0);

      const Type *T = QT.getTypePtr();
      if (T->isArrayType())
        T = T->getBaseElementTypeUnsafe();

      const CXXRecordDecl *Record = T->getAsCXXRecordDecl();
      if (!Record)
        return ScopePair(diag::note_protected_by_variable_init, 0);

      // If we need to call a non trivial destructor for this variable,
      // record an out diagnostic.
      unsigned OutDiag = 0;
      if (!Init->isGLValue() && !Record->hasTrivialDestructor())
        OutDiag = diag::note_exits_dtor;

      if (const CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(Init)) {
        const CXXConstructorDecl *ctor = cce->getConstructor();
        if (ctor->isTrivial() && ctor->isDefaultConstructor()) {
          if (OutDiag)
            InDiag = diag::note_protected_by_variable_nontriv_destructor;
          else if (!Record->isPOD())
            InDiag = diag::note_protected_by_variable_non_pod;
          return ScopePair(InDiag, OutDiag);
        }
      }

      return ScopePair(diag::note_protected_by_variable_init, OutDiag);
    }

    return ScopePair(InDiag, 0);
  }

  if (const TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return ScopePair(diag::note_protected_by_vla_typedef, 0);
  }

  if (const TypeAliasDecl *TD = dyn_cast<TypeAliasDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return ScopePair(diag::note_protected_by_vla_type_alias, 0);
  }

  return ScopePair(0U, 0U);
}

/// \brief Build scope information for a declaration that is part of a DeclStmt.
void JumpScopeChecker::BuildScopeInformation(Decl *D, unsigned &ParentScope) {
  // If this decl causes a new scope, push and switch to it.
  std::pair<unsigned,unsigned> Diags = GetDiagForGotoScopeDecl(S.Context, D);
  if (Diags.first || Diags.second) {
    Scopes.push_back(GotoScope(ParentScope, Diags.first, Diags.second,
                               D->getLocation()));
    ParentScope = Scopes.size()-1;
  }
  
  // If the decl has an initializer, walk it with the potentially new
  // scope we just installed.
  if (VarDecl *VD = dyn_cast<VarDecl>(D))
    if (Expr *Init = VD->getInit())
      BuildScopeInformation(Init, ParentScope);
}

/// \brief Build scope information for a captured block literal variables.
void JumpScopeChecker::BuildScopeInformation(VarDecl *D, 
                                             const BlockDecl *BDecl, 
                                             unsigned &ParentScope) {
  // exclude captured __block variables; there's no destructor
  // associated with the block literal for them.
  if (D->hasAttr<BlocksAttr>())
    return;
  QualType T = D->getType();
  QualType::DestructionKind destructKind = T.isDestructedType();
  if (destructKind != QualType::DK_none) {
    std::pair<unsigned,unsigned> Diags;
    switch (destructKind) {
      case QualType::DK_cxx_destructor:
        Diags = ScopePair(diag::note_enters_block_captures_cxx_obj,
                          diag::note_exits_block_captures_cxx_obj);
        break;
      case QualType::DK_objc_strong_lifetime:
        Diags = ScopePair(diag::note_enters_block_captures_strong,
                          diag::note_exits_block_captures_strong);
        break;
      case QualType::DK_objc_weak_lifetime:
        Diags = ScopePair(diag::note_enters_block_captures_weak,
                          diag::note_exits_block_captures_weak);
        break;
      case QualType::DK_none:
        llvm_unreachable("non-lifetime captured variable");
    }
    SourceLocation Loc = D->getLocation();
    if (Loc.isInvalid())
      Loc = BDecl->getLocation();
    Scopes.push_back(GotoScope(ParentScope, 
                               Diags.first, Diags.second, Loc));
    ParentScope = Scopes.size()-1;
  }
}

/// BuildScopeInformation - The statements from CI to CE are known to form a
/// coherent VLA scope with a specified parent node.  Walk through the
/// statements, adding any labels or gotos to LabelAndGotoScopes and recursively
/// walking the AST as needed.
void JumpScopeChecker::BuildScopeInformation(Stmt *S, unsigned &origParentScope) {
  // If this is a statement, rather than an expression, scopes within it don't
  // propagate out into the enclosing scope.  Otherwise we have to worry
  // about block literals, which have the lifetime of their enclosing statement.
  unsigned independentParentScope = origParentScope;
  unsigned &ParentScope = ((isa<Expr>(S) && !isa<StmtExpr>(S)) 
                            ? origParentScope : independentParentScope);

  bool SkipFirstSubStmt = false;
  
  // If we found a label, remember that it is in ParentScope scope.
  switch (S->getStmtClass()) {
  case Stmt::AddrLabelExprClass:
    IndirectJumpTargets.push_back(cast<AddrLabelExpr>(S)->getLabel());
    break;

  case Stmt::IndirectGotoStmtClass:
    // "goto *&&lbl;" is a special case which we treat as equivalent
    // to a normal goto.  In addition, we don't calculate scope in the
    // operand (to avoid recording the address-of-label use), which
    // works only because of the restricted set of expressions which
    // we detect as constant targets.
    if (cast<IndirectGotoStmt>(S)->getConstantTarget()) {
      LabelAndGotoScopes[S] = ParentScope;
      Jumps.push_back(S);
      return;
    }

    LabelAndGotoScopes[S] = ParentScope;
    IndirectJumps.push_back(cast<IndirectGotoStmt>(S));
    break;

  case Stmt::SwitchStmtClass:
    // Evaluate the condition variable before entering the scope of the switch
    // statement.
    if (VarDecl *Var = cast<SwitchStmt>(S)->getConditionVariable()) {
      BuildScopeInformation(Var, ParentScope);
      SkipFirstSubStmt = true;
    }
    // Fall through
      
  case Stmt::GotoStmtClass:
    // Remember both what scope a goto is in as well as the fact that we have
    // it.  This makes the second scan not have to walk the AST again.
    LabelAndGotoScopes[S] = ParentScope;
    Jumps.push_back(S);
    break;

  case Stmt::CXXTryStmtClass: {
    CXXTryStmt *TS = cast<CXXTryStmt>(S);
    unsigned newParentScope;
    Scopes.push_back(GotoScope(ParentScope,
                               diag::note_protected_by_cxx_try,
                               diag::note_exits_cxx_try,
                               TS->getSourceRange().getBegin()));
    if (Stmt *TryBlock = TS->getTryBlock())
      BuildScopeInformation(TryBlock, (newParentScope = Scopes.size()-1));

    // Jump from the catch into the try is not allowed either.
    for (unsigned I = 0, E = TS->getNumHandlers(); I != E; ++I) {
      CXXCatchStmt *CS = TS->getHandler(I);
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_cxx_catch,
                                 diag::note_exits_cxx_catch,
                                 CS->getSourceRange().getBegin()));
      BuildScopeInformation(CS->getHandlerBlock(), 
                            (newParentScope = Scopes.size()-1));
    }
    return;
  }

  default:
    break;
  }

  for (Stmt::child_range CI = S->children(); CI; ++CI) {
    if (SkipFirstSubStmt) {
      SkipFirstSubStmt = false;
      continue;
    }
    
    Stmt *SubStmt = *CI;
    if (SubStmt == 0) continue;

    // Cases, labels, and defaults aren't "scope parents".  It's also
    // important to handle these iteratively instead of recursively in
    // order to avoid blowing out the stack.
    while (true) {
      Stmt *Next;
      if (CaseStmt *CS = dyn_cast<CaseStmt>(SubStmt))
        Next = CS->getSubStmt();
      else if (DefaultStmt *DS = dyn_cast<DefaultStmt>(SubStmt))
        Next = DS->getSubStmt();
      else if (LabelStmt *LS = dyn_cast<LabelStmt>(SubStmt))
        Next = LS->getSubStmt();
      else
        break;

      LabelAndGotoScopes[SubStmt] = ParentScope;
      SubStmt = Next;
    }

    // If this is a declstmt with a VLA definition, it defines a scope from here
    // to the end of the containing context.
    if (DeclStmt *DS = dyn_cast<DeclStmt>(SubStmt)) {
      // The decl statement creates a scope if any of the decls in it are VLAs
      // or have the cleanup attribute.
      for (DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
           I != E; ++I)
        BuildScopeInformation(*I, ParentScope);
      continue;
    }
    // Disallow jumps into any part of an @try statement by pushing a scope and
    // walking all sub-stmts in that scope.
    if (ObjCAtTryStmt *AT = dyn_cast<ObjCAtTryStmt>(SubStmt)) {
      unsigned newParentScope;
      // Recursively walk the AST for the @try part.
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_objc_try,
                                 diag::note_exits_objc_try,
                                 AT->getAtTryLoc()));
      if (Stmt *TryPart = AT->getTryBody())
        BuildScopeInformation(TryPart, (newParentScope = Scopes.size()-1));

      // Jump from the catch to the finally or try is not valid.
      for (unsigned I = 0, N = AT->getNumCatchStmts(); I != N; ++I) {
        ObjCAtCatchStmt *AC = AT->getCatchStmt(I);
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_catch,
                                   diag::note_exits_objc_catch,
                                   AC->getAtCatchLoc()));
        // @catches are nested and it isn't
        BuildScopeInformation(AC->getCatchBody(), 
                              (newParentScope = Scopes.size()-1));
      }

      // Jump from the finally to the try or catch is not valid.
      if (ObjCAtFinallyStmt *AF = AT->getFinallyStmt()) {
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_finally,
                                   diag::note_exits_objc_finally,
                                   AF->getAtFinallyLoc()));
        BuildScopeInformation(AF, (newParentScope = Scopes.size()-1));
      }

      continue;
    }
    
    unsigned newParentScope;
    // Disallow jumps into the protected statement of an @synchronized, but
    // allow jumps into the object expression it protects.
    if (ObjCAtSynchronizedStmt *AS = dyn_cast<ObjCAtSynchronizedStmt>(SubStmt)){
      // Recursively walk the AST for the @synchronized object expr, it is
      // evaluated in the normal scope.
      BuildScopeInformation(AS->getSynchExpr(), ParentScope);

      // Recursively walk the AST for the @synchronized part, protected by a new
      // scope.
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_objc_synchronized,
                                 diag::note_exits_objc_synchronized,
                                 AS->getAtSynchronizedLoc()));
      BuildScopeInformation(AS->getSynchBody(), 
                            (newParentScope = Scopes.size()-1));
      continue;
    }

    // Disallow jumps into the protected statement of an @autoreleasepool.
    if (ObjCAutoreleasePoolStmt *AS = dyn_cast<ObjCAutoreleasePoolStmt>(SubStmt)){
      // Recursively walk the AST for the @autoreleasepool part, protected by a new
      // scope.
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_objc_autoreleasepool,
                                 diag::note_exits_objc_autoreleasepool,
                                 AS->getAtLoc()));
      BuildScopeInformation(AS->getSubStmt(), (newParentScope = Scopes.size()-1));
      continue;
    }

    // Disallow jumps past full-expressions that use blocks with
    // non-trivial cleanups of their captures.  This is theoretically
    // implementable but a lot of work which we haven't felt up to doing.
    if (ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(SubStmt)) {
      for (unsigned i = 0, e = EWC->getNumObjects(); i != e; ++i) {
        const BlockDecl *BDecl = EWC->getObject(i);
        for (BlockDecl::capture_const_iterator ci = BDecl->capture_begin(),
             ce = BDecl->capture_end(); ci != ce; ++ci) {
          VarDecl *variable = ci->getVariable();
          BuildScopeInformation(variable, BDecl, ParentScope);
        }
      }
    }
    
    // Recursively walk the AST.
    BuildScopeInformation(SubStmt, ParentScope);
  }
}

/// VerifyJumps - Verify each element of the Jumps array to see if they are
/// valid, emitting diagnostics if not.
void JumpScopeChecker::VerifyJumps() {
  while (!Jumps.empty()) {
    Stmt *Jump = Jumps.pop_back_val();

    // With a goto,
    if (GotoStmt *GS = dyn_cast<GotoStmt>(Jump)) {
      CheckJump(GS, GS->getLabel()->getStmt(), GS->getGotoLoc(),
                diag::err_goto_into_protected_scope,
                diag::warn_goto_into_protected_scope,
                diag::warn_cxx98_compat_goto_into_protected_scope);
      continue;
    }

    // We only get indirect gotos here when they have a constant target.
    if (IndirectGotoStmt *IGS = dyn_cast<IndirectGotoStmt>(Jump)) {
      LabelDecl *Target = IGS->getConstantTarget();
      CheckJump(IGS, Target->getStmt(), IGS->getGotoLoc(),
                diag::err_goto_into_protected_scope,
                diag::warn_goto_into_protected_scope,
                diag::warn_cxx98_compat_goto_into_protected_scope);
      continue;
    }

    SwitchStmt *SS = cast<SwitchStmt>(Jump);
    for (SwitchCase *SC = SS->getSwitchCaseList(); SC;
         SC = SC->getNextSwitchCase()) {
      assert(LabelAndGotoScopes.count(SC) && "Case not visited?");
      CheckJump(SS, SC, SC->getLocStart(),
                diag::err_switch_into_protected_scope, 0,
                diag::warn_cxx98_compat_switch_into_protected_scope);
    }
  }
}

/// VerifyIndirectJumps - Verify whether any possible indirect jump
/// might cross a protection boundary.  Unlike direct jumps, indirect
/// jumps count cleanups as protection boundaries:  since there's no
/// way to know where the jump is going, we can't implicitly run the
/// right cleanups the way we can with direct jumps.
///
/// Thus, an indirect jump is "trivial" if it bypasses no
/// initializations and no teardowns.  More formally, an indirect jump
/// from A to B is trivial if the path out from A to DCA(A,B) is
/// trivial and the path in from DCA(A,B) to B is trivial, where
/// DCA(A,B) is the deepest common ancestor of A and B.
/// Jump-triviality is transitive but asymmetric.
///
/// A path in is trivial if none of the entered scopes have an InDiag.
/// A path out is trivial is none of the exited scopes have an OutDiag.
///
/// Under these definitions, this function checks that the indirect
/// jump between A and B is trivial for every indirect goto statement A
/// and every label B whose address was taken in the function.
void JumpScopeChecker::VerifyIndirectJumps() {
  if (IndirectJumps.empty()) return;

  // If there aren't any address-of-label expressions in this function,
  // complain about the first indirect goto.
  if (IndirectJumpTargets.empty()) {
    S.Diag(IndirectJumps[0]->getGotoLoc(),
           diag::err_indirect_goto_without_addrlabel);
    return;
  }

  // Collect a single representative of every scope containing an
  // indirect goto.  For most code bases, this substantially cuts
  // down on the number of jump sites we'll have to consider later.
  typedef std::pair<unsigned, IndirectGotoStmt*> JumpScope;
  SmallVector<JumpScope, 32> JumpScopes;
  {
    llvm::DenseMap<unsigned, IndirectGotoStmt*> JumpScopesMap;
    for (SmallVectorImpl<IndirectGotoStmt*>::iterator
           I = IndirectJumps.begin(), E = IndirectJumps.end(); I != E; ++I) {
      IndirectGotoStmt *IG = *I;
      assert(LabelAndGotoScopes.count(IG) &&
             "indirect jump didn't get added to scopes?");
      unsigned IGScope = LabelAndGotoScopes[IG];
      IndirectGotoStmt *&Entry = JumpScopesMap[IGScope];
      if (!Entry) Entry = IG;
    }
    JumpScopes.reserve(JumpScopesMap.size());
    for (llvm::DenseMap<unsigned, IndirectGotoStmt*>::iterator
           I = JumpScopesMap.begin(), E = JumpScopesMap.end(); I != E; ++I)
      JumpScopes.push_back(*I);
  }

  // Collect a single representative of every scope containing a
  // label whose address was taken somewhere in the function.
  // For most code bases, there will be only one such scope.
  llvm::DenseMap<unsigned, LabelDecl*> TargetScopes;
  for (SmallVectorImpl<LabelDecl*>::iterator
         I = IndirectJumpTargets.begin(), E = IndirectJumpTargets.end();
       I != E; ++I) {
    LabelDecl *TheLabel = *I;
    assert(LabelAndGotoScopes.count(TheLabel->getStmt()) &&
           "Referenced label didn't get added to scopes?");
    unsigned LabelScope = LabelAndGotoScopes[TheLabel->getStmt()];
    LabelDecl *&Target = TargetScopes[LabelScope];
    if (!Target) Target = TheLabel;
  }

  // For each target scope, make sure it's trivially reachable from
  // every scope containing a jump site.
  //
  // A path between scopes always consists of exitting zero or more
  // scopes, then entering zero or more scopes.  We build a set of
  // of scopes S from which the target scope can be trivially
  // entered, then verify that every jump scope can be trivially
  // exitted to reach a scope in S.
  llvm::BitVector Reachable(Scopes.size(), false);
  for (llvm::DenseMap<unsigned,LabelDecl*>::iterator
         TI = TargetScopes.begin(), TE = TargetScopes.end(); TI != TE; ++TI) {
    unsigned TargetScope = TI->first;
    LabelDecl *TargetLabel = TI->second;

    Reachable.reset();

    // Mark all the enclosing scopes from which you can safely jump
    // into the target scope.  'Min' will end up being the index of
    // the shallowest such scope.
    unsigned Min = TargetScope;
    while (true) {
      Reachable.set(Min);

      // Don't go beyond the outermost scope.
      if (Min == 0) break;

      // Stop if we can't trivially enter the current scope.
      if (Scopes[Min].InDiag) break;

      Min = Scopes[Min].ParentScope;
    }

    // Walk through all the jump sites, checking that they can trivially
    // reach this label scope.
    for (SmallVectorImpl<JumpScope>::iterator
           I = JumpScopes.begin(), E = JumpScopes.end(); I != E; ++I) {
      unsigned Scope = I->first;

      // Walk out the "scope chain" for this scope, looking for a scope
      // we've marked reachable.  For well-formed code this amortizes
      // to O(JumpScopes.size() / Scopes.size()):  we only iterate
      // when we see something unmarked, and in well-formed code we
      // mark everything we iterate past.
      bool IsReachable = false;
      while (true) {
        if (Reachable.test(Scope)) {
          // If we find something reachable, mark all the scopes we just
          // walked through as reachable.
          for (unsigned S = I->first; S != Scope; S = Scopes[S].ParentScope)
            Reachable.set(S);
          IsReachable = true;
          break;
        }

        // Don't walk out if we've reached the top-level scope or we've
        // gotten shallower than the shallowest reachable scope.
        if (Scope == 0 || Scope < Min) break;

        // Don't walk out through an out-diagnostic.
        if (Scopes[Scope].OutDiag) break;

        Scope = Scopes[Scope].ParentScope;
      }

      // Only diagnose if we didn't find something.
      if (IsReachable) continue;

      DiagnoseIndirectJump(I->second, I->first, TargetLabel, TargetScope);
    }
  }
}

/// Return true if a particular error+note combination must be downgraded to a
/// warning in Microsoft mode.
static bool IsMicrosoftJumpWarning(unsigned JumpDiag, unsigned InDiagNote) {
  return (JumpDiag == diag::err_goto_into_protected_scope &&
         (InDiagNote == diag::note_protected_by_variable_init ||
          InDiagNote == diag::note_protected_by_variable_nontriv_destructor));
}

/// Return true if a particular note should be downgraded to a compatibility
/// warning in C++11 mode.
static bool IsCXX98CompatWarning(Sema &S, unsigned InDiagNote) {
  return S.getLangOpts().CPlusPlus0x &&
         InDiagNote == diag::note_protected_by_variable_non_pod;
}

/// Produce primary diagnostic for an indirect jump statement.
static void DiagnoseIndirectJumpStmt(Sema &S, IndirectGotoStmt *Jump,
                                     LabelDecl *Target, bool &Diagnosed) {
  if (Diagnosed)
    return;
  S.Diag(Jump->getGotoLoc(), diag::err_indirect_goto_in_protected_scope);
  S.Diag(Target->getStmt()->getIdentLoc(), diag::note_indirect_goto_target);
  Diagnosed = true;
}

/// Produce note diagnostics for a jump into a protected scope.
void JumpScopeChecker::NoteJumpIntoScopes(ArrayRef<unsigned> ToScopes) {
  assert(!ToScopes.empty());
  for (unsigned I = 0, E = ToScopes.size(); I != E; ++I)
    if (Scopes[ToScopes[I]].InDiag)
      S.Diag(Scopes[ToScopes[I]].Loc, Scopes[ToScopes[I]].InDiag);
}

/// Diagnose an indirect jump which is known to cross scopes.
void JumpScopeChecker::DiagnoseIndirectJump(IndirectGotoStmt *Jump,
                                            unsigned JumpScope,
                                            LabelDecl *Target,
                                            unsigned TargetScope) {
  assert(JumpScope != TargetScope);

  unsigned Common = GetDeepestCommonScope(JumpScope, TargetScope);
  bool Diagnosed = false;

  // Walk out the scope chain until we reach the common ancestor.
  for (unsigned I = JumpScope; I != Common; I = Scopes[I].ParentScope)
    if (Scopes[I].OutDiag) {
      DiagnoseIndirectJumpStmt(S, Jump, Target, Diagnosed);
      S.Diag(Scopes[I].Loc, Scopes[I].OutDiag);
    }

  SmallVector<unsigned, 10> ToScopesCXX98Compat;

  // Now walk into the scopes containing the label whose address was taken.
  for (unsigned I = TargetScope; I != Common; I = Scopes[I].ParentScope)
    if (IsCXX98CompatWarning(S, Scopes[I].InDiag))
      ToScopesCXX98Compat.push_back(I);
    else if (Scopes[I].InDiag) {
      DiagnoseIndirectJumpStmt(S, Jump, Target, Diagnosed);
      S.Diag(Scopes[I].Loc, Scopes[I].InDiag);
    }

  // Diagnose this jump if it would be ill-formed in C++98.
  if (!Diagnosed && !ToScopesCXX98Compat.empty()) {
    S.Diag(Jump->getGotoLoc(),
           diag::warn_cxx98_compat_indirect_goto_in_protected_scope);
    S.Diag(Target->getStmt()->getIdentLoc(), diag::note_indirect_goto_target);
    NoteJumpIntoScopes(ToScopesCXX98Compat);
  }
}

/// CheckJump - Validate that the specified jump statement is valid: that it is
/// jumping within or out of its current scope, not into a deeper one.
void JumpScopeChecker::CheckJump(Stmt *From, Stmt *To, SourceLocation DiagLoc,
                               unsigned JumpDiagError, unsigned JumpDiagWarning,
                                 unsigned JumpDiagCXX98Compat) {
  assert(LabelAndGotoScopes.count(From) && "Jump didn't get added to scopes?");
  unsigned FromScope = LabelAndGotoScopes[From];

  assert(LabelAndGotoScopes.count(To) && "Jump didn't get added to scopes?");
  unsigned ToScope = LabelAndGotoScopes[To];

  // Common case: exactly the same scope, which is fine.
  if (FromScope == ToScope) return;

  unsigned CommonScope = GetDeepestCommonScope(FromScope, ToScope);

  // It's okay to jump out from a nested scope.
  if (CommonScope == ToScope) return;

  // Pull out (and reverse) any scopes we might need to diagnose skipping.
  SmallVector<unsigned, 10> ToScopesCXX98Compat;
  SmallVector<unsigned, 10> ToScopesError;
  SmallVector<unsigned, 10> ToScopesWarning;
  for (unsigned I = ToScope; I != CommonScope; I = Scopes[I].ParentScope) {
    if (S.getLangOpts().MicrosoftMode && JumpDiagWarning != 0 &&
        IsMicrosoftJumpWarning(JumpDiagError, Scopes[I].InDiag))
      ToScopesWarning.push_back(I);
    else if (IsCXX98CompatWarning(S, Scopes[I].InDiag))
      ToScopesCXX98Compat.push_back(I);
    else if (Scopes[I].InDiag)
      ToScopesError.push_back(I);
  }

  // Handle warnings.
  if (!ToScopesWarning.empty()) {
    S.Diag(DiagLoc, JumpDiagWarning);
    NoteJumpIntoScopes(ToScopesWarning);
  }

  // Handle errors.
  if (!ToScopesError.empty()) {
    S.Diag(DiagLoc, JumpDiagError);
    NoteJumpIntoScopes(ToScopesError);
  }

  // Handle -Wc++98-compat warnings if the jump is well-formed.
  if (ToScopesError.empty() && !ToScopesCXX98Compat.empty()) {
    S.Diag(DiagLoc, JumpDiagCXX98Compat);
    NoteJumpIntoScopes(ToScopesCXX98Compat);
  }
}

void Sema::DiagnoseInvalidJumps(Stmt *Body) {
  (void)JumpScopeChecker(Body, *this);
}
