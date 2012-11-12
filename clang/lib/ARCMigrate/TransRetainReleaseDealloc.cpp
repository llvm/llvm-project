//===--- TransRetainReleaseDealloc.cpp - Tranformations to ARC mode -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// removeRetainReleaseDealloc:
//
// Removes retain/release/autorelease/dealloc messages.
//
//  return [[foo retain] autorelease];
// ---->
//  return foo;
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

class RetainReleaseDeallocRemover :
                       public RecursiveASTVisitor<RetainReleaseDeallocRemover> {
  Stmt *Body;
  MigrationPass &Pass;

  ExprSet Removables;
  OwningPtr<ParentMap> StmtMap;

  Selector DelegateSel, FinalizeSel;

public:
  RetainReleaseDeallocRemover(MigrationPass &pass)
    : Body(0), Pass(pass) {
    DelegateSel =
        Pass.Ctx.Selectors.getNullarySelector(&Pass.Ctx.Idents.get("delegate"));
    FinalizeSel =
        Pass.Ctx.Selectors.getNullarySelector(&Pass.Ctx.Idents.get("finalize"));
  }

  void transformBody(Stmt *body, Decl *ParentD) {
    Body = body;
    collectRemovables(body, Removables);
    StmtMap.reset(new ParentMap(body));
    TraverseStmt(body);
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    switch (E->getMethodFamily()) {
    default:
      if (E->isInstanceMessage() && E->getSelector() == FinalizeSel)
        break;
      return true;
    case OMF_autorelease:
      if (isRemovable(E)) {
        if (!isCommonUnusedAutorelease(E)) {
          // An unused autorelease is badness. If we remove it the receiver
          // will likely die immediately while previously it was kept alive
          // by the autorelease pool. This is bad practice in general, leave it
          // and emit an error to force the user to restructure his code.
          Pass.TA.reportError("it is not safe to remove an unused 'autorelease' "
              "message; its receiver may be destroyed immediately",
              E->getLocStart(), E->getSourceRange());
          return true;
        }
      }
      // Pass through.
    case OMF_retain:
    case OMF_release:
      if (E->getReceiverKind() == ObjCMessageExpr::Instance)
        if (Expr *rec = E->getInstanceReceiver()) {
          rec = rec->IgnoreParenImpCasts();
          if (rec->getType().getObjCLifetime() == Qualifiers::OCL_ExplicitNone &&
              (E->getMethodFamily() != OMF_retain || isRemovable(E))) {
            std::string err = "it is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "an __unsafe_unretained type";
            Pass.TA.reportError(err, rec->getLocStart());
            return true;
          }

          if (isGlobalVar(rec) &&
              (E->getMethodFamily() != OMF_retain || isRemovable(E))) {
            std::string err = "it is not safe to remove '";
            err += E->getSelector().getAsString() + "' message on "
                "a global variable";
            Pass.TA.reportError(err, rec->getLocStart());
            return true;
          }

          if (E->getMethodFamily() == OMF_release && isDelegateMessage(rec)) {
            Pass.TA.reportError("it is not safe to remove 'retain' "
                "message on the result of a 'delegate' message; "
                "the object that was passed to 'setDelegate:' may not be "
                "properly retained", rec->getLocStart());
            return true;
          }
        }
    case OMF_dealloc:
      break;
    }

    switch (E->getReceiverKind()) {
    default:
      return true;
    case ObjCMessageExpr::SuperInstance: {
      Transaction Trans(Pass.TA);
      clearDiagnostics(E->getSuperLoc());
      if (tryRemoving(E))
        return true;
      Pass.TA.replace(E->getSourceRange(), "self");
      return true;
    }
    case ObjCMessageExpr::Instance:
      break;
    }

    Expr *rec = E->getInstanceReceiver();
    if (!rec) return true;

    Transaction Trans(Pass.TA);
    clearDiagnostics(rec->getExprLoc());

    ObjCMessageExpr *Msg = E;
    Expr *RecContainer = Msg;
    SourceRange RecRange = rec->getSourceRange();
    checkForGCDOrXPC(Msg, RecContainer, rec, RecRange);

    if (Msg->getMethodFamily() == OMF_release &&
        isRemovable(RecContainer) && isInAtFinally(RecContainer)) {
      // Change the -release to "receiver = nil" in a finally to avoid a leak
      // when an exception is thrown.
      Pass.TA.replace(RecContainer->getSourceRange(), RecRange);
      std::string str = " = ";
      str += getNilString(Pass.Ctx);
      Pass.TA.insertAfterToken(RecRange.getEnd(), str);
      return true;
    }

    if (!hasSideEffects(rec, Pass.Ctx)) {
      if (tryRemoving(RecContainer))
        return true;
    }
    Pass.TA.replace(RecContainer->getSourceRange(), RecRange);

    return true;
  }

private:
  /// \brief Checks for idioms where an unused -autorelease is common.
  ///
  /// Currently only returns true for this idiom which is common in property
  /// setters:
  ///
  ///   [backingValue autorelease];
  ///   backingValue = [newValue retain]; // in general a +1 assign
  ///
  bool isCommonUnusedAutorelease(ObjCMessageExpr *E) {
    Expr *Rec = E->getInstanceReceiver();
    if (!Rec)
      return false;

    Decl *RefD = getReferencedDecl(Rec);
    if (!RefD)
      return false;

    Stmt *OuterS = E, *InnerS;
    do {
      InnerS = OuterS;
      OuterS = StmtMap->getParent(InnerS);
    }
    while (OuterS && (isa<ParenExpr>(OuterS) ||
                      isa<CastExpr>(OuterS) ||
                      isa<ExprWithCleanups>(OuterS)));
    
    if (!OuterS)
      return false;

    // Find next statement after the -autorelease.

    Stmt::child_iterator currChildS = OuterS->child_begin();
    Stmt::child_iterator childE = OuterS->child_end();
    for (; currChildS != childE; ++currChildS) {
      if (*currChildS == InnerS)
        break;
    }
    if (currChildS == childE)
      return false;
    ++currChildS;
    if (currChildS == childE)
      return false;

    Stmt *nextStmt = *currChildS;
    if (!nextStmt)
      return false;
    nextStmt = nextStmt->IgnoreImplicit();

    // Check for "RefD = [+1 retained object];".
    
    if (BinaryOperator *Bop = dyn_cast<BinaryOperator>(nextStmt)) {
      if (RefD != getReferencedDecl(Bop->getLHS()))
        return false;
      if (isPlusOneAssign(Bop))
        return true;
    }
    return false;
  }

  Decl *getReferencedDecl(Expr *E) {
    if (!E)
      return 0;

    E = E->IgnoreParenCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
      return DRE->getDecl();
    if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
      return ME->getMemberDecl();
    if (ObjCIvarRefExpr *IRE = dyn_cast<ObjCIvarRefExpr>(E))
      return IRE->getDecl();

    return 0;
  }

  /// \brief Check if the retain/release is due to a GCD/XPC macro that are
  /// defined as:
  ///
  /// #define dispatch_retain(object) ({ dispatch_object_t _o = (object); _dispatch_object_validate(_o); (void)[_o retain]; })
  /// #define dispatch_release(object) ({ dispatch_object_t _o = (object); _dispatch_object_validate(_o); [_o release]; })
  /// #define xpc_retain(object) ({ xpc_object_t _o = (object); _xpc_object_validate(_o); [_o retain]; })
  /// #define xpc_release(object) ({ xpc_object_t _o = (object); _xpc_object_validate(_o); [_o release]; })
  ///
  /// and return the top container which is the StmtExpr and the macro argument
  /// expression.
  void checkForGCDOrXPC(ObjCMessageExpr *Msg, Expr *&RecContainer,
                        Expr *&Rec, SourceRange &RecRange) {
    SourceLocation Loc = Msg->getExprLoc();
    if (!Loc.isMacroID())
      return;
    SourceManager &SM = Pass.Ctx.getSourceManager();
    StringRef MacroName = Lexer::getImmediateMacroName(Loc, SM,
                                                     Pass.Ctx.getLangOpts());
    bool isGCDOrXPC = llvm::StringSwitch<bool>(MacroName)
        .Case("dispatch_retain", true)
        .Case("dispatch_release", true)
        .Case("xpc_retain", true)
        .Case("xpc_release", true)
        .Default(false);
    if (!isGCDOrXPC)
      return;

    StmtExpr *StmtE = 0;
    Stmt *S = Msg;
    while (S) {
      if (StmtExpr *SE = dyn_cast<StmtExpr>(S)) {
        StmtE = SE;
        break;
      }
      S = StmtMap->getParent(S);
    }

    if (!StmtE)
      return;

    Stmt::child_range StmtExprChild = StmtE->children();
    if (!StmtExprChild)
      return;
    CompoundStmt *CompS = dyn_cast_or_null<CompoundStmt>(*StmtExprChild);
    if (!CompS)
      return;

    Stmt::child_range CompStmtChild = CompS->children();
    if (!CompStmtChild)
      return;
    DeclStmt *DeclS = dyn_cast_or_null<DeclStmt>(*CompStmtChild);
    if (!DeclS)
      return;
    if (!DeclS->isSingleDecl())
      return;
    VarDecl *VD = dyn_cast_or_null<VarDecl>(DeclS->getSingleDecl());
    if (!VD)
      return;
    Expr *Init = VD->getInit();
    if (!Init)
      return;

    RecContainer = StmtE;
    Rec = Init->IgnoreParenImpCasts();
    if (ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(Rec))
      Rec = EWC->getSubExpr()->IgnoreParenImpCasts();
    RecRange = Rec->getSourceRange();
    if (SM.isMacroArgExpansion(RecRange.getBegin()))
      RecRange.setBegin(SM.getImmediateSpellingLoc(RecRange.getBegin()));
    if (SM.isMacroArgExpansion(RecRange.getEnd()))
      RecRange.setEnd(SM.getImmediateSpellingLoc(RecRange.getEnd()));
  }

  void clearDiagnostics(SourceLocation loc) const {
    Pass.TA.clearDiagnostic(diag::err_arc_illegal_explicit_message,
                            diag::err_unavailable,
                            diag::err_unavailable_message,
                            loc);
  }

  bool isDelegateMessage(Expr *E) const {
    if (!E) return false;

    E = E->IgnoreParenCasts();

    // Also look through property-getter sugar.
    if (PseudoObjectExpr *pseudoOp = dyn_cast<PseudoObjectExpr>(E))
      E = pseudoOp->getResultExpr()->IgnoreImplicit();

    if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(E))
      return (ME->isInstanceMessage() && ME->getSelector() == DelegateSel);

    return false;
  }

  bool isInAtFinally(Expr *E) const {
    assert(E);
    Stmt *S = E;
    while (S) {
      if (isa<ObjCAtFinallyStmt>(S))
        return true;
      S = StmtMap->getParent(S);
    }

    return false;
  }

  bool isRemovable(Expr *E) const {
    return Removables.count(E);
  }
  
  bool tryRemoving(Expr *E) const {
    if (isRemovable(E)) {
      Pass.TA.removeStmt(E);
      return true;
    }

    Stmt *parent = StmtMap->getParent(E);

    if (ImplicitCastExpr *castE = dyn_cast_or_null<ImplicitCastExpr>(parent))
      return tryRemoving(castE);

    if (ParenExpr *parenE = dyn_cast_or_null<ParenExpr>(parent))
      return tryRemoving(parenE);

    if (BinaryOperator *
          bopE = dyn_cast_or_null<BinaryOperator>(parent)) {
      if (bopE->getOpcode() == BO_Comma && bopE->getLHS() == E &&
          isRemovable(bopE)) {
        Pass.TA.replace(bopE->getSourceRange(), bopE->getRHS()->getSourceRange());
        return true;
      }
    }

    return false;
  }

};

} // anonymous namespace

void trans::removeRetainReleaseDeallocFinalize(MigrationPass &pass) {
  BodyTransform<RetainReleaseDeallocRemover> trans(pass);
  trans.TraverseDecl(pass.Ctx.getTranslationUnitDecl());
}
