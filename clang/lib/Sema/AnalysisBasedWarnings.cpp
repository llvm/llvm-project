//=== AnalysisBasedWarnings.cpp - Sema warnings based on libAnalysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines analysis_warnings::[Policy,Executor].
// Together they are used by Sema to issue warnings based on inexpensive
// static analysis algorithms in libAnalysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/Analyses/CalledOnceCheck.h"
#include "clang/Analysis/Analyses/Consumed.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/Analyses/ThreadSafety.h"
#include "clang/Analysis/Analyses/UninitializedValues.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <deque>
#include <iterator>
#include <optional>

using namespace clang;

//===----------------------------------------------------------------------===//
// Unreachable code analysis.
//===----------------------------------------------------------------------===//

namespace {
  class UnreachableCodeHandler : public reachable_code::Callback {
    Sema &S;
    SourceRange PreviousSilenceableCondVal;

  public:
    UnreachableCodeHandler(Sema &s) : S(s) {}

    void HandleUnreachable(reachable_code::UnreachableKind UK, SourceLocation L,
                           SourceRange SilenceableCondVal, SourceRange R1,
                           SourceRange R2, bool HasFallThroughAttr) override {
      // If the diagnosed code is `[[fallthrough]];` and
      // `-Wunreachable-code-fallthrough` is  enabled, suppress `code will never
      // be executed` warning to avoid generating diagnostic twice
      if (HasFallThroughAttr &&
          !S.getDiagnostics().isIgnored(diag::warn_unreachable_fallthrough_attr,
                                        SourceLocation()))
        return;

      // Avoid reporting multiple unreachable code diagnostics that are
      // triggered by the same conditional value.
      if (PreviousSilenceableCondVal.isValid() &&
          SilenceableCondVal.isValid() &&
          PreviousSilenceableCondVal == SilenceableCondVal)
        return;
      PreviousSilenceableCondVal = SilenceableCondVal;

      unsigned diag = diag::warn_unreachable;
      switch (UK) {
        case reachable_code::UK_Break:
          diag = diag::warn_unreachable_break;
          break;
        case reachable_code::UK_Return:
          diag = diag::warn_unreachable_return;
          break;
        case reachable_code::UK_Loop_Increment:
          diag = diag::warn_unreachable_loop_increment;
          break;
        case reachable_code::UK_Other:
          break;
      }

      S.Diag(L, diag) << R1 << R2;

      SourceLocation Open = SilenceableCondVal.getBegin();
      if (Open.isValid()) {
        SourceLocation Close = SilenceableCondVal.getEnd();
        Close = S.getLocForEndOfToken(Close);
        if (Close.isValid()) {
          S.Diag(Open, diag::note_unreachable_silence)
            << FixItHint::CreateInsertion(Open, "/* DISABLES CODE */ (")
            << FixItHint::CreateInsertion(Close, ")");
        }
      }
    }
  };
} // anonymous namespace

/// CheckUnreachable - Check for unreachable code.
static void CheckUnreachable(Sema &S, AnalysisDeclContext &AC) {
  // As a heuristic prune all diagnostics not in the main file.  Currently
  // the majority of warnings in headers are false positives.  These
  // are largely caused by configuration state, e.g. preprocessor
  // defined code, etc.
  //
  // Note that this is also a performance optimization.  Analyzing
  // headers many times can be expensive.
  if (!S.getSourceManager().isInMainFile(AC.getDecl()->getBeginLoc()))
    return;

  UnreachableCodeHandler UC(S);
  reachable_code::FindUnreachableCode(AC, S.getPreprocessor(), UC);
}

namespace {
/// Warn on logical operator errors in CFGBuilder
class LogicalErrorHandler : public CFGCallback {
  Sema &S;

public:
  LogicalErrorHandler(Sema &S) : S(S) {}

  static bool HasMacroID(const Expr *E) {
    if (E->getExprLoc().isMacroID())
      return true;

    // Recurse to children.
    for (const Stmt *SubStmt : E->children())
      if (const Expr *SubExpr = dyn_cast_or_null<Expr>(SubStmt))
        if (HasMacroID(SubExpr))
          return true;

    return false;
  }

  void logicAlwaysTrue(const BinaryOperator *B, bool isAlwaysTrue) override {
    if (HasMacroID(B))
      return;

    unsigned DiagID = isAlwaysTrue
                          ? diag::warn_tautological_negation_or_compare
                          : diag::warn_tautological_negation_and_compare;
    SourceRange DiagRange = B->getSourceRange();
    S.Diag(B->getExprLoc(), DiagID) << DiagRange;
  }

  void compareAlwaysTrue(const BinaryOperator *B, bool isAlwaysTrue) override {
    if (HasMacroID(B))
      return;

    SourceRange DiagRange = B->getSourceRange();
    S.Diag(B->getExprLoc(), diag::warn_tautological_overlap_comparison)
        << DiagRange << isAlwaysTrue;
  }

  void compareBitwiseEquality(const BinaryOperator *B,
                              bool isAlwaysTrue) override {
    if (HasMacroID(B))
      return;

    SourceRange DiagRange = B->getSourceRange();
    S.Diag(B->getExprLoc(), diag::warn_comparison_bitwise_always)
        << DiagRange << isAlwaysTrue;
  }

  void compareBitwiseOr(const BinaryOperator *B) override {
    if (HasMacroID(B))
      return;

    SourceRange DiagRange = B->getSourceRange();
    S.Diag(B->getExprLoc(), diag::warn_comparison_bitwise_or) << DiagRange;
  }

  static bool hasActiveDiagnostics(DiagnosticsEngine &Diags,
                                   SourceLocation Loc) {
    return !Diags.isIgnored(diag::warn_tautological_overlap_comparison, Loc) ||
           !Diags.isIgnored(diag::warn_comparison_bitwise_or, Loc) ||
           !Diags.isIgnored(diag::warn_tautological_negation_and_compare, Loc);
  }
};
} // anonymous namespace

//===----------------------------------------------------------------------===//
// Check for infinite self-recursion in functions
//===----------------------------------------------------------------------===//

// Returns true if the function is called anywhere within the CFGBlock.
// For member functions, the additional condition of being call from the
// this pointer is required.
static bool hasRecursiveCallInPath(const FunctionDecl *FD, CFGBlock &Block) {
  // Process all the Stmt's in this block to find any calls to FD.
  for (const auto &B : Block) {
    if (B.getKind() != CFGElement::Statement)
      continue;

    const CallExpr *CE = dyn_cast<CallExpr>(B.getAs<CFGStmt>()->getStmt());
    if (!CE || !CE->getCalleeDecl() ||
        CE->getCalleeDecl()->getCanonicalDecl() != FD)
      continue;

    // Skip function calls which are qualified with a templated class.
    if (const DeclRefExpr *DRE =
            dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreParenImpCasts())) {
      if (NestedNameSpecifier *NNS = DRE->getQualifier()) {
        if (NNS->getKind() == NestedNameSpecifier::TypeSpec &&
            isa<TemplateSpecializationType>(NNS->getAsType())) {
          continue;
        }
      }
    }

    const CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(CE);
    if (!MCE || isa<CXXThisExpr>(MCE->getImplicitObjectArgument()) ||
        !MCE->getMethodDecl()->isVirtual())
      return true;
  }
  return false;
}

// Returns true if every path from the entry block passes through a call to FD.
static bool checkForRecursiveFunctionCall(const FunctionDecl *FD, CFG *cfg) {
  llvm::SmallPtrSet<CFGBlock *, 16> Visited;
  llvm::SmallVector<CFGBlock *, 16> WorkList;
  // Keep track of whether we found at least one recursive path.
  bool foundRecursion = false;

  const unsigned ExitID = cfg->getExit().getBlockID();

  // Seed the work list with the entry block.
  WorkList.push_back(&cfg->getEntry());

  while (!WorkList.empty()) {
    CFGBlock *Block = WorkList.pop_back_val();

    for (auto I = Block->succ_begin(), E = Block->succ_end(); I != E; ++I) {
      if (CFGBlock *SuccBlock = *I) {
        if (!Visited.insert(SuccBlock).second)
          continue;

        // Found a path to the exit node without a recursive call.
        if (ExitID == SuccBlock->getBlockID())
          return false;

        // If the successor block contains a recursive call, end analysis there.
        if (hasRecursiveCallInPath(FD, *SuccBlock)) {
          foundRecursion = true;
          continue;
        }

        WorkList.push_back(SuccBlock);
      }
    }
  }
  return foundRecursion;
}

static void checkRecursiveFunction(Sema &S, const FunctionDecl *FD,
                                   const Stmt *Body, AnalysisDeclContext &AC) {
  FD = FD->getCanonicalDecl();

  // Only run on non-templated functions and non-templated members of
  // templated classes.
  if (FD->getTemplatedKind() != FunctionDecl::TK_NonTemplate &&
      FD->getTemplatedKind() != FunctionDecl::TK_MemberSpecialization)
    return;

  CFG *cfg = AC.getCFG();
  if (!cfg) return;

  // If the exit block is unreachable, skip processing the function.
  if (cfg->getExit().pred_empty())
    return;

  // Emit diagnostic if a recursive function call is detected for all paths.
  if (checkForRecursiveFunctionCall(FD, cfg))
    S.Diag(Body->getBeginLoc(), diag::warn_infinite_recursive_function);
}

//===----------------------------------------------------------------------===//
// Check for throw in a non-throwing function.
//===----------------------------------------------------------------------===//

/// Determine whether an exception thrown by E, unwinding from ThrowBlock,
/// can reach ExitBlock.
static bool throwEscapes(Sema &S, const CXXThrowExpr *E, CFGBlock &ThrowBlock,
                         CFG *Body) {
  SmallVector<CFGBlock *, 16> Stack;
  llvm::BitVector Queued(Body->getNumBlockIDs());

  Stack.push_back(&ThrowBlock);
  Queued[ThrowBlock.getBlockID()] = true;

  while (!Stack.empty()) {
    CFGBlock &UnwindBlock = *Stack.back();
    Stack.pop_back();

    for (auto &Succ : UnwindBlock.succs()) {
      if (!Succ.isReachable() || Queued[Succ->getBlockID()])
        continue;

      if (Succ->getBlockID() == Body->getExit().getBlockID())
        return true;

      if (auto *Catch =
              dyn_cast_or_null<CXXCatchStmt>(Succ->getLabel())) {
        QualType Caught = Catch->getCaughtType();
        if (Caught.isNull() || // catch (...) catches everything
            !E->getSubExpr() || // throw; is considered cuaght by any handler
            S.handlerCanCatch(Caught, E->getSubExpr()->getType()))
          // Exception doesn't escape via this path.
          break;
      } else {
        Stack.push_back(Succ);
        Queued[Succ->getBlockID()] = true;
      }
    }
  }

  return false;
}

static void visitReachableThrows(
    CFG *BodyCFG,
    llvm::function_ref<void(const CXXThrowExpr *, CFGBlock &)> Visit) {
  llvm::BitVector Reachable(BodyCFG->getNumBlockIDs());
  clang::reachable_code::ScanReachableFromBlock(&BodyCFG->getEntry(), Reachable);
  for (CFGBlock *B : *BodyCFG) {
    if (!Reachable[B->getBlockID()])
      continue;
    for (CFGElement &E : *B) {
      std::optional<CFGStmt> S = E.getAs<CFGStmt>();
      if (!S)
        continue;
      if (auto *Throw = dyn_cast<CXXThrowExpr>(S->getStmt()))
        Visit(Throw, *B);
    }
  }
}

static void EmitDiagForCXXThrowInNonThrowingFunc(Sema &S, SourceLocation OpLoc,
                                                 const FunctionDecl *FD) {
  if (!S.getSourceManager().isInSystemHeader(OpLoc) &&
      FD->getTypeSourceInfo()) {
    S.Diag(OpLoc, diag::warn_throw_in_noexcept_func) << FD;
    if (S.getLangOpts().CPlusPlus11 &&
        (isa<CXXDestructorDecl>(FD) ||
         FD->getDeclName().getCXXOverloadedOperator() == OO_Delete ||
         FD->getDeclName().getCXXOverloadedOperator() == OO_Array_Delete)) {
      if (const auto *Ty = FD->getTypeSourceInfo()->getType()->
                                         getAs<FunctionProtoType>())
        S.Diag(FD->getLocation(), diag::note_throw_in_dtor)
            << !isa<CXXDestructorDecl>(FD) << !Ty->hasExceptionSpec()
            << FD->getExceptionSpecSourceRange();
    } else
      S.Diag(FD->getLocation(), diag::note_throw_in_function)
          << FD->getExceptionSpecSourceRange();
  }
}

static void checkThrowInNonThrowingFunc(Sema &S, const FunctionDecl *FD,
                                        AnalysisDeclContext &AC) {
  CFG *BodyCFG = AC.getCFG();
  if (!BodyCFG)
    return;
  if (BodyCFG->getExit().pred_empty())
    return;
  visitReachableThrows(BodyCFG, [&](const CXXThrowExpr *Throw, CFGBlock &Block) {
    if (throwEscapes(S, Throw, Block, BodyCFG))
      EmitDiagForCXXThrowInNonThrowingFunc(S, Throw->getThrowLoc(), FD);
  });
}

static bool isNoexcept(const FunctionDecl *FD) {
  const auto *FPT = FD->getType()->castAs<FunctionProtoType>();
  if (FPT->isNothrow() || FD->hasAttr<NoThrowAttr>())
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Check for missing return value.
//===----------------------------------------------------------------------===//

enum ControlFlowKind {
  UnknownFallThrough,
  NeverFallThrough,
  MaybeFallThrough,
  AlwaysFallThrough,
  NeverFallThroughOrReturn
};

/// CheckFallThrough - Check that we don't fall off the end of a
/// Statement that should return a value.
///
/// \returns AlwaysFallThrough iff we always fall off the end of the statement,
/// MaybeFallThrough iff we might or might not fall off the end,
/// NeverFallThroughOrReturn iff we never fall off the end of the statement or
/// return.  We assume NeverFallThrough iff we never fall off the end of the
/// statement but we may return.  We assume that functions not marked noreturn
/// will return.
static ControlFlowKind CheckFallThrough(AnalysisDeclContext &AC) {
  CFG *cfg = AC.getCFG();
  if (!cfg) return UnknownFallThrough;

  // The CFG leaves in dead things, and we don't want the dead code paths to
  // confuse us, so we mark all live things first.
  llvm::BitVector live(cfg->getNumBlockIDs());
  unsigned count = reachable_code::ScanReachableFromBlock(&cfg->getEntry(),
                                                          live);

  bool AddEHEdges = AC.getAddEHEdges();
  if (!AddEHEdges && count != cfg->getNumBlockIDs())
    // When there are things remaining dead, and we didn't add EH edges
    // from CallExprs to the catch clauses, we have to go back and
    // mark them as live.
    for (const auto *B : *cfg) {
      if (!live[B->getBlockID()]) {
        if (B->pred_begin() == B->pred_end()) {
          const Stmt *Term = B->getTerminatorStmt();
          if (isa_and_nonnull<CXXTryStmt>(Term))
            // When not adding EH edges from calls, catch clauses
            // can otherwise seem dead.  Avoid noting them as dead.
            count += reachable_code::ScanReachableFromBlock(B, live);
          continue;
        }
      }
    }

  // Now we know what is live, we check the live precessors of the exit block
  // and look for fall through paths, being careful to ignore normal returns,
  // and exceptional paths.
  bool HasLiveReturn = false;
  bool HasFakeEdge = false;
  bool HasPlainEdge = false;
  bool HasAbnormalEdge = false;

  // Ignore default cases that aren't likely to be reachable because all
  // enums in a switch(X) have explicit case statements.
  CFGBlock::FilterOptions FO;
  FO.IgnoreDefaultsWithCoveredEnums = 1;

  for (CFGBlock::filtered_pred_iterator I =
           cfg->getExit().filtered_pred_start_end(FO);
       I.hasMore(); ++I) {
    const CFGBlock &B = **I;
    if (!live[B.getBlockID()])
      continue;

    // Skip blocks which contain an element marked as no-return. They don't
    // represent actually viable edges into the exit block, so mark them as
    // abnormal.
    if (B.hasNoReturnElement()) {
      HasAbnormalEdge = true;
      continue;
    }

    // Destructors can appear after the 'return' in the CFG.  This is
    // normal.  We need to look pass the destructors for the return
    // statement (if it exists).
    CFGBlock::const_reverse_iterator ri = B.rbegin(), re = B.rend();

    for ( ; ri != re ; ++ri)
      if (ri->getAs<CFGStmt>())
        break;

    // No more CFGElements in the block?
    if (ri == re) {
      const Stmt *Term = B.getTerminatorStmt();
      if (Term && (isa<CXXTryStmt>(Term) || isa<ObjCAtTryStmt>(Term))) {
        HasAbnormalEdge = true;
        continue;
      }
      // A labeled empty statement, or the entry block...
      HasPlainEdge = true;
      continue;
    }

    CFGStmt CS = ri->castAs<CFGStmt>();
    const Stmt *S = CS.getStmt();
    if (isa<ReturnStmt>(S) || isa<CoreturnStmt>(S)) {
      HasLiveReturn = true;
      continue;
    }
    if (isa<ObjCAtThrowStmt>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (isa<CXXThrowExpr>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (isa<MSAsmStmt>(S)) {
      // TODO: Verify this is correct.
      HasFakeEdge = true;
      HasLiveReturn = true;
      continue;
    }
    if (isa<CXXTryStmt>(S)) {
      HasAbnormalEdge = true;
      continue;
    }
    if (!llvm::is_contained(B.succs(), &cfg->getExit())) {
      HasAbnormalEdge = true;
      continue;
    }

    HasPlainEdge = true;
  }
  if (!HasPlainEdge) {
    if (HasLiveReturn)
      return NeverFallThrough;
    return NeverFallThroughOrReturn;
  }
  if (HasAbnormalEdge || HasFakeEdge || HasLiveReturn)
    return MaybeFallThrough;
  // This says AlwaysFallThrough for calls to functions that are not marked
  // noreturn, that don't return.  If people would like this warning to be more
  // accurate, such functions should be marked as noreturn.
  return AlwaysFallThrough;
}

namespace {

struct CheckFallThroughDiagnostics {
  unsigned diag_MaybeFallThrough_HasNoReturn;
  unsigned diag_MaybeFallThrough_ReturnsNonVoid;
  unsigned diag_AlwaysFallThrough_HasNoReturn;
  unsigned diag_AlwaysFallThrough_ReturnsNonVoid;
  unsigned diag_NeverFallThroughOrReturn;
  enum { Function, Block, Lambda, Coroutine } funMode;
  SourceLocation FuncLoc;

  static CheckFallThroughDiagnostics MakeForFunction(const Decl *Func) {
    CheckFallThroughDiagnostics D;
    D.FuncLoc = Func->getLocation();
    D.diag_MaybeFallThrough_HasNoReturn =
      diag::warn_falloff_noreturn_function;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
      diag::warn_maybe_falloff_nonvoid_function;
    D.diag_AlwaysFallThrough_HasNoReturn =
      diag::warn_falloff_noreturn_function;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
      diag::warn_falloff_nonvoid_function;

    // Don't suggest that virtual functions be marked "noreturn", since they
    // might be overridden by non-noreturn functions.
    bool isVirtualMethod = false;
    if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Func))
      isVirtualMethod = Method->isVirtual();

    // Don't suggest that template instantiations be marked "noreturn"
    bool isTemplateInstantiation = false;
    if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(Func))
      isTemplateInstantiation = Function->isTemplateInstantiation();

    if (!isVirtualMethod && !isTemplateInstantiation)
      D.diag_NeverFallThroughOrReturn =
        diag::warn_suggest_noreturn_function;
    else
      D.diag_NeverFallThroughOrReturn = 0;

    D.funMode = Function;
    return D;
  }

  static CheckFallThroughDiagnostics MakeForCoroutine(const Decl *Func) {
    CheckFallThroughDiagnostics D;
    D.FuncLoc = Func->getLocation();
    D.diag_MaybeFallThrough_HasNoReturn = 0;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
        diag::warn_maybe_falloff_nonvoid_coroutine;
    D.diag_AlwaysFallThrough_HasNoReturn = 0;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
        diag::warn_falloff_nonvoid_coroutine;
    D.diag_NeverFallThroughOrReturn = 0;
    D.funMode = Coroutine;
    return D;
  }

  static CheckFallThroughDiagnostics MakeForBlock() {
    CheckFallThroughDiagnostics D;
    D.diag_MaybeFallThrough_HasNoReturn =
      diag::err_noreturn_block_has_return_expr;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
      diag::err_maybe_falloff_nonvoid_block;
    D.diag_AlwaysFallThrough_HasNoReturn =
      diag::err_noreturn_block_has_return_expr;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
      diag::err_falloff_nonvoid_block;
    D.diag_NeverFallThroughOrReturn = 0;
    D.funMode = Block;
    return D;
  }

  static CheckFallThroughDiagnostics MakeForLambda() {
    CheckFallThroughDiagnostics D;
    D.diag_MaybeFallThrough_HasNoReturn =
      diag::err_noreturn_lambda_has_return_expr;
    D.diag_MaybeFallThrough_ReturnsNonVoid =
      diag::warn_maybe_falloff_nonvoid_lambda;
    D.diag_AlwaysFallThrough_HasNoReturn =
      diag::err_noreturn_lambda_has_return_expr;
    D.diag_AlwaysFallThrough_ReturnsNonVoid =
      diag::warn_falloff_nonvoid_lambda;
    D.diag_NeverFallThroughOrReturn = 0;
    D.funMode = Lambda;
    return D;
  }

  bool checkDiagnostics(DiagnosticsEngine &D, bool ReturnsVoid,
                        bool HasNoReturn) const {
    if (funMode == Function) {
      return (ReturnsVoid ||
              D.isIgnored(diag::warn_maybe_falloff_nonvoid_function,
                          FuncLoc)) &&
             (!HasNoReturn ||
              D.isIgnored(diag::warn_noreturn_function_has_return_expr,
                          FuncLoc)) &&
             (!ReturnsVoid ||
              D.isIgnored(diag::warn_suggest_noreturn_block, FuncLoc));
    }
    if (funMode == Coroutine) {
      return (ReturnsVoid ||
              D.isIgnored(diag::warn_maybe_falloff_nonvoid_function, FuncLoc) ||
              D.isIgnored(diag::warn_maybe_falloff_nonvoid_coroutine,
                          FuncLoc)) &&
             (!HasNoReturn);
    }
    // For blocks / lambdas.
    return ReturnsVoid && !HasNoReturn;
  }
};

} // anonymous namespace

/// CheckFallThroughForBody - Check that we don't fall off the end of a
/// function that should return a value.  Check that we don't fall off the end
/// of a noreturn function.  We assume that functions and blocks not marked
/// noreturn will return.
static void CheckFallThroughForBody(Sema &S, const Decl *D, const Stmt *Body,
                                    QualType BlockType,
                                    const CheckFallThroughDiagnostics &CD,
                                    AnalysisDeclContext &AC,
                                    sema::FunctionScopeInfo *FSI) {

  bool ReturnsVoid = false;
  bool HasNoReturn = false;
  bool IsCoroutine = FSI->isCoroutine();

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (const auto *CBody = dyn_cast<CoroutineBodyStmt>(Body))
      ReturnsVoid = CBody->getFallthroughHandler() != nullptr;
    else
      ReturnsVoid = FD->getReturnType()->isVoidType();
    HasNoReturn = FD->isNoReturn();
  }
  else if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
    ReturnsVoid = MD->getReturnType()->isVoidType();
    HasNoReturn = MD->hasAttr<NoReturnAttr>();
  }
  else if (isa<BlockDecl>(D)) {
    if (const FunctionType *FT =
          BlockType->getPointeeType()->getAs<FunctionType>()) {
      if (FT->getReturnType()->isVoidType())
        ReturnsVoid = true;
      if (FT->getNoReturnAttr())
        HasNoReturn = true;
    }
  }

  DiagnosticsEngine &Diags = S.getDiagnostics();

  // Short circuit for compilation speed.
  if (CD.checkDiagnostics(Diags, ReturnsVoid, HasNoReturn))
      return;
  SourceLocation LBrace = Body->getBeginLoc(), RBrace = Body->getEndLoc();
  auto EmitDiag = [&](SourceLocation Loc, unsigned DiagID) {
    if (IsCoroutine)
      S.Diag(Loc, DiagID) << FSI->CoroutinePromise->getType();
    else
      S.Diag(Loc, DiagID);
  };

  // cpu_dispatch functions permit empty function bodies for ICC compatibility.
  if (D->getAsFunction() && D->getAsFunction()->isCPUDispatchMultiVersion())
    return;

  // Either in a function body compound statement, or a function-try-block.
  switch (CheckFallThrough(AC)) {
    case UnknownFallThrough:
      break;

    case MaybeFallThrough:
      if (HasNoReturn)
        EmitDiag(RBrace, CD.diag_MaybeFallThrough_HasNoReturn);
      else if (!ReturnsVoid)
        EmitDiag(RBrace, CD.diag_MaybeFallThrough_ReturnsNonVoid);
      break;
    case AlwaysFallThrough:
      if (HasNoReturn)
        EmitDiag(RBrace, CD.diag_AlwaysFallThrough_HasNoReturn);
      else if (!ReturnsVoid)
        EmitDiag(RBrace, CD.diag_AlwaysFallThrough_ReturnsNonVoid);
      break;
    case NeverFallThroughOrReturn:
      if (ReturnsVoid && !HasNoReturn && CD.diag_NeverFallThroughOrReturn) {
        if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
          S.Diag(LBrace, CD.diag_NeverFallThroughOrReturn) << 0 << FD;
        } else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
          S.Diag(LBrace, CD.diag_NeverFallThroughOrReturn) << 1 << MD;
        } else {
          S.Diag(LBrace, CD.diag_NeverFallThroughOrReturn);
        }
      }
      break;
    case NeverFallThrough:
      break;
  }
}

//===----------------------------------------------------------------------===//
// -Wuninitialized
//===----------------------------------------------------------------------===//

namespace {
/// ContainsReference - A visitor class to search for references to
/// a particular declaration (the needle) within any evaluated component of an
/// expression (recursively).
class ContainsReference : public ConstEvaluatedExprVisitor<ContainsReference> {
  bool FoundReference;
  const DeclRefExpr *Needle;

public:
  typedef ConstEvaluatedExprVisitor<ContainsReference> Inherited;

  ContainsReference(ASTContext &Context, const DeclRefExpr *Needle)
    : Inherited(Context), FoundReference(false), Needle(Needle) {}

  void VisitExpr(const Expr *E) {
    // Stop evaluating if we already have a reference.
    if (FoundReference)
      return;

    Inherited::VisitExpr(E);
  }

  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (E == Needle)
      FoundReference = true;
    else
      Inherited::VisitDeclRefExpr(E);
  }

  bool doesContainReference() const { return FoundReference; }
};
} // anonymous namespace

static bool SuggestInitializationFixit(Sema &S, const VarDecl *VD) {
  QualType VariableTy = VD->getType().getCanonicalType();
  if (VariableTy->isBlockPointerType() &&
      !VD->hasAttr<BlocksAttr>()) {
    S.Diag(VD->getLocation(), diag::note_block_var_fixit_add_initialization)
        << VD->getDeclName()
        << FixItHint::CreateInsertion(VD->getLocation(), "__block ");
    return true;
  }

  // Don't issue a fixit if there is already an initializer.
  if (VD->getInit())
    return false;

  // Don't suggest a fixit inside macros.
  if (VD->getEndLoc().isMacroID())
    return false;

  SourceLocation Loc = S.getLocForEndOfToken(VD->getEndLoc());

  // Suggest possible initialization (if any).
  std::string Init = S.getFixItZeroInitializerForType(VariableTy, Loc);
  if (Init.empty())
    return false;

  S.Diag(Loc, diag::note_var_fixit_add_initialization) << VD->getDeclName()
    << FixItHint::CreateInsertion(Loc, Init);
  return true;
}

/// Create a fixit to remove an if-like statement, on the assumption that its
/// condition is CondVal.
static void CreateIfFixit(Sema &S, const Stmt *If, const Stmt *Then,
                          const Stmt *Else, bool CondVal,
                          FixItHint &Fixit1, FixItHint &Fixit2) {
  if (CondVal) {
    // If condition is always true, remove all but the 'then'.
    Fixit1 = FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(If->getBeginLoc(), Then->getBeginLoc()));
    if (Else) {
      SourceLocation ElseKwLoc = S.getLocForEndOfToken(Then->getEndLoc());
      Fixit2 =
          FixItHint::CreateRemoval(SourceRange(ElseKwLoc, Else->getEndLoc()));
    }
  } else {
    // If condition is always false, remove all but the 'else'.
    if (Else)
      Fixit1 = FixItHint::CreateRemoval(CharSourceRange::getCharRange(
          If->getBeginLoc(), Else->getBeginLoc()));
    else
      Fixit1 = FixItHint::CreateRemoval(If->getSourceRange());
  }
}

/// DiagUninitUse -- Helper function to produce a diagnostic for an
/// uninitialized use of a variable.
static void DiagUninitUse(Sema &S, const VarDecl *VD, const UninitUse &Use,
                          bool IsCapturedByBlock) {
  bool Diagnosed = false;

  switch (Use.getKind()) {
  case UninitUse::Always:
    S.Diag(Use.getUser()->getBeginLoc(), diag::warn_uninit_var)
        << VD->getDeclName() << IsCapturedByBlock
        << Use.getUser()->getSourceRange();
    return;

  case UninitUse::AfterDecl:
  case UninitUse::AfterCall:
    S.Diag(VD->getLocation(), diag::warn_sometimes_uninit_var)
      << VD->getDeclName() << IsCapturedByBlock
      << (Use.getKind() == UninitUse::AfterDecl ? 4 : 5)
      << const_cast<DeclContext*>(VD->getLexicalDeclContext())
      << VD->getSourceRange();
    S.Diag(Use.getUser()->getBeginLoc(), diag::note_uninit_var_use)
        << IsCapturedByBlock << Use.getUser()->getSourceRange();
    return;

  case UninitUse::Maybe:
  case UninitUse::Sometimes:
    // Carry on to report sometimes-uninitialized branches, if possible,
    // or a 'may be used uninitialized' diagnostic otherwise.
    break;
  }

  // Diagnose each branch which leads to a sometimes-uninitialized use.
  for (UninitUse::branch_iterator I = Use.branch_begin(), E = Use.branch_end();
       I != E; ++I) {
    assert(Use.getKind() == UninitUse::Sometimes);

    const Expr *User = Use.getUser();
    const Stmt *Term = I->Terminator;

    // Information used when building the diagnostic.
    unsigned DiagKind;
    StringRef Str;
    SourceRange Range;

    // FixIts to suppress the diagnostic by removing the dead condition.
    // For all binary terminators, branch 0 is taken if the condition is true,
    // and branch 1 is taken if the condition is false.
    int RemoveDiagKind = -1;
    const char *FixitStr =
        S.getLangOpts().CPlusPlus ? (I->Output ? "true" : "false")
                                  : (I->Output ? "1" : "0");
    FixItHint Fixit1, Fixit2;

    switch (Term ? Term->getStmtClass() : Stmt::DeclStmtClass) {
    default:
      // Don't know how to report this. Just fall back to 'may be used
      // uninitialized'. FIXME: Can this happen?
      continue;

    // "condition is true / condition is false".
    case Stmt::IfStmtClass: {
      const IfStmt *IS = cast<IfStmt>(Term);
      DiagKind = 0;
      Str = "if";
      Range = IS->getCond()->getSourceRange();
      RemoveDiagKind = 0;
      CreateIfFixit(S, IS, IS->getThen(), IS->getElse(),
                    I->Output, Fixit1, Fixit2);
      break;
    }
    case Stmt::ConditionalOperatorClass: {
      const ConditionalOperator *CO = cast<ConditionalOperator>(Term);
      DiagKind = 0;
      Str = "?:";
      Range = CO->getCond()->getSourceRange();
      RemoveDiagKind = 0;
      CreateIfFixit(S, CO, CO->getTrueExpr(), CO->getFalseExpr(),
                    I->Output, Fixit1, Fixit2);
      break;
    }
    case Stmt::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(Term);
      if (!BO->isLogicalOp())
        continue;
      DiagKind = 0;
      Str = BO->getOpcodeStr();
      Range = BO->getLHS()->getSourceRange();
      RemoveDiagKind = 0;
      if ((BO->getOpcode() == BO_LAnd && I->Output) ||
          (BO->getOpcode() == BO_LOr && !I->Output))
        // true && y -> y, false || y -> y.
        Fixit1 = FixItHint::CreateRemoval(
            SourceRange(BO->getBeginLoc(), BO->getOperatorLoc()));
      else
        // false && y -> false, true || y -> true.
        Fixit1 = FixItHint::CreateReplacement(BO->getSourceRange(), FixitStr);
      break;
    }

    // "loop is entered / loop is exited".
    case Stmt::WhileStmtClass:
      DiagKind = 1;
      Str = "while";
      Range = cast<WhileStmt>(Term)->getCond()->getSourceRange();
      RemoveDiagKind = 1;
      Fixit1 = FixItHint::CreateReplacement(Range, FixitStr);
      break;
    case Stmt::ForStmtClass:
      DiagKind = 1;
      Str = "for";
      Range = cast<ForStmt>(Term)->getCond()->getSourceRange();
      RemoveDiagKind = 1;
      if (I->Output)
        Fixit1 = FixItHint::CreateRemoval(Range);
      else
        Fixit1 = FixItHint::CreateReplacement(Range, FixitStr);
      break;
    case Stmt::CXXForRangeStmtClass:
      if (I->Output == 1) {
        // The use occurs if a range-based for loop's body never executes.
        // That may be impossible, and there's no syntactic fix for this,
        // so treat it as a 'may be uninitialized' case.
        continue;
      }
      DiagKind = 1;
      Str = "for";
      Range = cast<CXXForRangeStmt>(Term)->getRangeInit()->getSourceRange();
      break;

    // "condition is true / loop is exited".
    case Stmt::DoStmtClass:
      DiagKind = 2;
      Str = "do";
      Range = cast<DoStmt>(Term)->getCond()->getSourceRange();
      RemoveDiagKind = 1;
      Fixit1 = FixItHint::CreateReplacement(Range, FixitStr);
      break;

    // "switch case is taken".
    case Stmt::CaseStmtClass:
      DiagKind = 3;
      Str = "case";
      Range = cast<CaseStmt>(Term)->getLHS()->getSourceRange();
      break;
    case Stmt::DefaultStmtClass:
      DiagKind = 3;
      Str = "default";
      Range = cast<DefaultStmt>(Term)->getDefaultLoc();
      break;
    }

    S.Diag(Range.getBegin(), diag::warn_sometimes_uninit_var)
      << VD->getDeclName() << IsCapturedByBlock << DiagKind
      << Str << I->Output << Range;
    S.Diag(User->getBeginLoc(), diag::note_uninit_var_use)
        << IsCapturedByBlock << User->getSourceRange();
    if (RemoveDiagKind != -1)
      S.Diag(Fixit1.RemoveRange.getBegin(), diag::note_uninit_fixit_remove_cond)
        << RemoveDiagKind << Str << I->Output << Fixit1 << Fixit2;

    Diagnosed = true;
  }

  if (!Diagnosed)
    S.Diag(Use.getUser()->getBeginLoc(), diag::warn_maybe_uninit_var)
        << VD->getDeclName() << IsCapturedByBlock
        << Use.getUser()->getSourceRange();
}

/// Diagnose uninitialized const reference usages.
static bool DiagnoseUninitializedConstRefUse(Sema &S, const VarDecl *VD,
                                             const UninitUse &Use) {
  S.Diag(Use.getUser()->getBeginLoc(), diag::warn_uninit_const_reference)
      << VD->getDeclName() << Use.getUser()->getSourceRange();
  return true;
}

/// DiagnoseUninitializedUse -- Helper function for diagnosing uses of an
/// uninitialized variable. This manages the different forms of diagnostic
/// emitted for particular types of uses. Returns true if the use was diagnosed
/// as a warning. If a particular use is one we omit warnings for, returns
/// false.
static bool DiagnoseUninitializedUse(Sema &S, const VarDecl *VD,
                                     const UninitUse &Use,
                                     bool alwaysReportSelfInit = false) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Use.getUser())) {
    // Inspect the initializer of the variable declaration which is
    // being referenced prior to its initialization. We emit
    // specialized diagnostics for self-initialization, and we
    // specifically avoid warning about self references which take the
    // form of:
    //
    //   int x = x;
    //
    // This is used to indicate to GCC that 'x' is intentionally left
    // uninitialized. Proven code paths which access 'x' in
    // an uninitialized state after this will still warn.
    if (const Expr *Initializer = VD->getInit()) {
      if (!alwaysReportSelfInit && DRE == Initializer->IgnoreParenImpCasts())
        return false;

      ContainsReference CR(S.Context, DRE);
      CR.Visit(Initializer);
      if (CR.doesContainReference()) {
        S.Diag(DRE->getBeginLoc(), diag::warn_uninit_self_reference_in_init)
            << VD->getDeclName() << VD->getLocation() << DRE->getSourceRange();
        return true;
      }
    }

    DiagUninitUse(S, VD, Use, false);
  } else {
    const BlockExpr *BE = cast<BlockExpr>(Use.getUser());
    if (VD->getType()->isBlockPointerType() && !VD->hasAttr<BlocksAttr>())
      S.Diag(BE->getBeginLoc(),
             diag::warn_uninit_byref_blockvar_captured_by_block)
          << VD->getDeclName()
          << VD->getType().getQualifiers().hasObjCLifetime();
    else
      DiagUninitUse(S, VD, Use, true);
  }

  // Report where the variable was declared when the use wasn't within
  // the initializer of that declaration & we didn't already suggest
  // an initialization fixit.
  if (!SuggestInitializationFixit(S, VD))
    S.Diag(VD->getBeginLoc(), diag::note_var_declared_here)
        << VD->getDeclName();

  return true;
}

namespace {
  class FallthroughMapper : public RecursiveASTVisitor<FallthroughMapper> {
  public:
    FallthroughMapper(Sema &S)
      : FoundSwitchStatements(false),
        S(S) {
    }

    bool foundSwitchStatements() const { return FoundSwitchStatements; }

    void markFallthroughVisited(const AttributedStmt *Stmt) {
      bool Found = FallthroughStmts.erase(Stmt);
      assert(Found);
      (void)Found;
    }

    typedef llvm::SmallPtrSet<const AttributedStmt*, 8> AttrStmts;

    const AttrStmts &getFallthroughStmts() const {
      return FallthroughStmts;
    }

    void fillReachableBlocks(CFG *Cfg) {
      assert(ReachableBlocks.empty() && "ReachableBlocks already filled");
      std::deque<const CFGBlock *> BlockQueue;

      ReachableBlocks.insert(&Cfg->getEntry());
      BlockQueue.push_back(&Cfg->getEntry());
      // Mark all case blocks reachable to avoid problems with switching on
      // constants, covered enums, etc.
      // These blocks can contain fall-through annotations, and we don't want to
      // issue a warn_fallthrough_attr_unreachable for them.
      for (const auto *B : *Cfg) {
        const Stmt *L = B->getLabel();
        if (isa_and_nonnull<SwitchCase>(L) && ReachableBlocks.insert(B).second)
          BlockQueue.push_back(B);
      }

      while (!BlockQueue.empty()) {
        const CFGBlock *P = BlockQueue.front();
        BlockQueue.pop_front();
        for (const CFGBlock *B : P->succs()) {
          if (B && ReachableBlocks.insert(B).second)
            BlockQueue.push_back(B);
        }
      }
    }

    bool checkFallThroughIntoBlock(const CFGBlock &B, int &AnnotatedCnt,
                                   bool IsTemplateInstantiation) {
      assert(!ReachableBlocks.empty() && "ReachableBlocks empty");

      int UnannotatedCnt = 0;
      AnnotatedCnt = 0;

      std::deque<const CFGBlock*> BlockQueue(B.pred_begin(), B.pred_end());
      while (!BlockQueue.empty()) {
        const CFGBlock *P = BlockQueue.front();
        BlockQueue.pop_front();
        if (!P) continue;

        const Stmt *Term = P->getTerminatorStmt();
        if (isa_and_nonnull<SwitchStmt>(Term))
          continue; // Switch statement, good.

        const SwitchCase *SW = dyn_cast_or_null<SwitchCase>(P->getLabel());
        if (SW && SW->getSubStmt() == B.getLabel() && P->begin() == P->end())
          continue; // Previous case label has no statements, good.

        const LabelStmt *L = dyn_cast_or_null<LabelStmt>(P->getLabel());
        if (L && L->getSubStmt() == B.getLabel() && P->begin() == P->end())
          continue; // Case label is preceded with a normal label, good.

        if (!ReachableBlocks.count(P)) {
          for (const CFGElement &Elem : llvm::reverse(*P)) {
            if (std::optional<CFGStmt> CS = Elem.getAs<CFGStmt>()) {
            if (const AttributedStmt *AS = asFallThroughAttr(CS->getStmt())) {
              // Don't issue a warning for an unreachable fallthrough
              // attribute in template instantiations as it may not be
              // unreachable in all instantiations of the template.
              if (!IsTemplateInstantiation)
                S.Diag(AS->getBeginLoc(),
                       diag::warn_unreachable_fallthrough_attr);
              markFallthroughVisited(AS);
              ++AnnotatedCnt;
              break;
            }
            // Don't care about other unreachable statements.
            }
          }
          // If there are no unreachable statements, this may be a special
          // case in CFG:
          // case X: {
          //    A a;  // A has a destructor.
          //    break;
          // }
          // // <<<< This place is represented by a 'hanging' CFG block.
          // case Y:
          continue;
        }

        const Stmt *LastStmt = getLastStmt(*P);
        if (const AttributedStmt *AS = asFallThroughAttr(LastStmt)) {
          markFallthroughVisited(AS);
          ++AnnotatedCnt;
          continue; // Fallthrough annotation, good.
        }

        if (!LastStmt) { // This block contains no executable statements.
          // Traverse its predecessors.
          std::copy(P->pred_begin(), P->pred_end(),
                    std::back_inserter(BlockQueue));
          continue;
        }

        ++UnannotatedCnt;
      }
      return !!UnannotatedCnt;
    }

    // RecursiveASTVisitor setup.
    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool VisitAttributedStmt(AttributedStmt *S) {
      if (asFallThroughAttr(S))
        FallthroughStmts.insert(S);
      return true;
    }

    bool VisitSwitchStmt(SwitchStmt *S) {
      FoundSwitchStatements = true;
      return true;
    }

    // We don't want to traverse local type declarations. We analyze their
    // methods separately.
    bool TraverseDecl(Decl *D) { return true; }

    // We analyze lambda bodies separately. Skip them here.
    bool TraverseLambdaExpr(LambdaExpr *LE) {
      // Traverse the captures, but not the body.
      for (const auto C : zip(LE->captures(), LE->capture_inits()))
        TraverseLambdaCapture(LE, &std::get<0>(C), std::get<1>(C));
      return true;
    }

  private:

    static const AttributedStmt *asFallThroughAttr(const Stmt *S) {
      if (const AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(S)) {
        if (hasSpecificAttr<FallThroughAttr>(AS->getAttrs()))
          return AS;
      }
      return nullptr;
    }

    static const Stmt *getLastStmt(const CFGBlock &B) {
      if (const Stmt *Term = B.getTerminatorStmt())
        return Term;
      for (const CFGElement &Elem : llvm::reverse(B))
        if (std::optional<CFGStmt> CS = Elem.getAs<CFGStmt>())
          return CS->getStmt();
      // Workaround to detect a statement thrown out by CFGBuilder:
      //   case X: {} case Y:
      //   case X: ; case Y:
      if (const SwitchCase *SW = dyn_cast_or_null<SwitchCase>(B.getLabel()))
        if (!isa<SwitchCase>(SW->getSubStmt()))
          return SW->getSubStmt();

      return nullptr;
    }

    bool FoundSwitchStatements;
    AttrStmts FallthroughStmts;
    Sema &S;
    llvm::SmallPtrSet<const CFGBlock *, 16> ReachableBlocks;
  };
} // anonymous namespace

static StringRef getFallthroughAttrSpelling(Preprocessor &PP,
                                            SourceLocation Loc) {
  TokenValue FallthroughTokens[] = {
    tok::l_square, tok::l_square,
    PP.getIdentifierInfo("fallthrough"),
    tok::r_square, tok::r_square
  };

  TokenValue ClangFallthroughTokens[] = {
    tok::l_square, tok::l_square, PP.getIdentifierInfo("clang"),
    tok::coloncolon, PP.getIdentifierInfo("fallthrough"),
    tok::r_square, tok::r_square
  };

  bool PreferClangAttr = !PP.getLangOpts().CPlusPlus17 && !PP.getLangOpts().C23;

  StringRef MacroName;
  if (PreferClangAttr)
    MacroName = PP.getLastMacroWithSpelling(Loc, ClangFallthroughTokens);
  if (MacroName.empty())
    MacroName = PP.getLastMacroWithSpelling(Loc, FallthroughTokens);
  if (MacroName.empty() && !PreferClangAttr)
    MacroName = PP.getLastMacroWithSpelling(Loc, ClangFallthroughTokens);
  if (MacroName.empty()) {
    if (!PreferClangAttr)
      MacroName = "[[fallthrough]]";
    else if (PP.getLangOpts().CPlusPlus)
      MacroName = "[[clang::fallthrough]]";
    else
      MacroName = "__attribute__((fallthrough))";
  }
  return MacroName;
}

static void DiagnoseSwitchLabelsFallthrough(Sema &S, AnalysisDeclContext &AC,
                                            bool PerFunction) {
  FallthroughMapper FM(S);
  FM.TraverseStmt(AC.getBody());

  if (!FM.foundSwitchStatements())
    return;

  if (PerFunction && FM.getFallthroughStmts().empty())
    return;

  CFG *Cfg = AC.getCFG();

  if (!Cfg)
    return;

  FM.fillReachableBlocks(Cfg);

  for (const CFGBlock *B : llvm::reverse(*Cfg)) {
    const Stmt *Label = B->getLabel();

    if (!isa_and_nonnull<SwitchCase>(Label))
      continue;

    int AnnotatedCnt;

    bool IsTemplateInstantiation = false;
    if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(AC.getDecl()))
      IsTemplateInstantiation = Function->isTemplateInstantiation();
    if (!FM.checkFallThroughIntoBlock(*B, AnnotatedCnt,
                                      IsTemplateInstantiation))
      continue;

    S.Diag(Label->getBeginLoc(),
           PerFunction ? diag::warn_unannotated_fallthrough_per_function
                       : diag::warn_unannotated_fallthrough);

    if (!AnnotatedCnt) {
      SourceLocation L = Label->getBeginLoc();
      if (L.isMacroID())
        continue;

      const Stmt *Term = B->getTerminatorStmt();
      // Skip empty cases.
      while (B->empty() && !Term && B->succ_size() == 1) {
        B = *B->succ_begin();
        Term = B->getTerminatorStmt();
      }
      if (!(B->empty() && isa_and_nonnull<BreakStmt>(Term))) {
        Preprocessor &PP = S.getPreprocessor();
        StringRef AnnotationSpelling = getFallthroughAttrSpelling(PP, L);
        SmallString<64> TextToInsert(AnnotationSpelling);
        TextToInsert += "; ";
        S.Diag(L, diag::note_insert_fallthrough_fixit)
            << AnnotationSpelling
            << FixItHint::CreateInsertion(L, TextToInsert);
      }
      S.Diag(L, diag::note_insert_break_fixit)
          << FixItHint::CreateInsertion(L, "break; ");
    }
  }

  for (const auto *F : FM.getFallthroughStmts())
    S.Diag(F->getBeginLoc(), diag::err_fallthrough_attr_invalid_placement);
}

static bool isInLoop(const ASTContext &Ctx, const ParentMap &PM,
                     const Stmt *S) {
  assert(S);

  do {
    switch (S->getStmtClass()) {
    case Stmt::ForStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::ObjCForCollectionStmtClass:
      return true;
    case Stmt::DoStmtClass: {
      Expr::EvalResult Result;
      if (!cast<DoStmt>(S)->getCond()->EvaluateAsInt(Result, Ctx))
        return true;
      return Result.Val.getInt().getBoolValue();
    }
    default:
      break;
    }
  } while ((S = PM.getParent(S)));

  return false;
}

static void diagnoseRepeatedUseOfWeak(Sema &S,
                                      const sema::FunctionScopeInfo *CurFn,
                                      const Decl *D,
                                      const ParentMap &PM) {
  typedef sema::FunctionScopeInfo::WeakObjectProfileTy WeakObjectProfileTy;
  typedef sema::FunctionScopeInfo::WeakObjectUseMap WeakObjectUseMap;
  typedef sema::FunctionScopeInfo::WeakUseVector WeakUseVector;
  typedef std::pair<const Stmt *, WeakObjectUseMap::const_iterator>
  StmtUsesPair;

  ASTContext &Ctx = S.getASTContext();

  const WeakObjectUseMap &WeakMap = CurFn->getWeakObjectUses();

  // Extract all weak objects that are referenced more than once.
  SmallVector<StmtUsesPair, 8> UsesByStmt;
  for (WeakObjectUseMap::const_iterator I = WeakMap.begin(), E = WeakMap.end();
       I != E; ++I) {
    const WeakUseVector &Uses = I->second;

    // Find the first read of the weak object.
    WeakUseVector::const_iterator UI = Uses.begin(), UE = Uses.end();
    for ( ; UI != UE; ++UI) {
      if (UI->isUnsafe())
        break;
    }

    // If there were only writes to this object, don't warn.
    if (UI == UE)
      continue;

    // If there was only one read, followed by any number of writes, and the
    // read is not within a loop, don't warn. Additionally, don't warn in a
    // loop if the base object is a local variable -- local variables are often
    // changed in loops.
    if (UI == Uses.begin()) {
      WeakUseVector::const_iterator UI2 = UI;
      for (++UI2; UI2 != UE; ++UI2)
        if (UI2->isUnsafe())
          break;

      if (UI2 == UE) {
        if (!isInLoop(Ctx, PM, UI->getUseExpr()))
          continue;

        const WeakObjectProfileTy &Profile = I->first;
        if (!Profile.isExactProfile())
          continue;

        const NamedDecl *Base = Profile.getBase();
        if (!Base)
          Base = Profile.getProperty();
        assert(Base && "A profile always has a base or property.");

        if (const VarDecl *BaseVar = dyn_cast<VarDecl>(Base))
          if (BaseVar->hasLocalStorage() && !isa<ParmVarDecl>(Base))
            continue;
      }
    }

    UsesByStmt.push_back(StmtUsesPair(UI->getUseExpr(), I));
  }

  if (UsesByStmt.empty())
    return;

  // Sort by first use so that we emit the warnings in a deterministic order.
  SourceManager &SM = S.getSourceManager();
  llvm::sort(UsesByStmt,
             [&SM](const StmtUsesPair &LHS, const StmtUsesPair &RHS) {
               return SM.isBeforeInTranslationUnit(LHS.first->getBeginLoc(),
                                                   RHS.first->getBeginLoc());
             });

  // Classify the current code body for better warning text.
  // This enum should stay in sync with the cases in
  // warn_arc_repeated_use_of_weak and warn_arc_possible_repeated_use_of_weak.
  // FIXME: Should we use a common classification enum and the same set of
  // possibilities all throughout Sema?
  enum {
    Function,
    Method,
    Block,
    Lambda
  } FunctionKind;

  if (isa<sema::BlockScopeInfo>(CurFn))
    FunctionKind = Block;
  else if (isa<sema::LambdaScopeInfo>(CurFn))
    FunctionKind = Lambda;
  else if (isa<ObjCMethodDecl>(D))
    FunctionKind = Method;
  else
    FunctionKind = Function;

  // Iterate through the sorted problems and emit warnings for each.
  for (const auto &P : UsesByStmt) {
    const Stmt *FirstRead = P.first;
    const WeakObjectProfileTy &Key = P.second->first;
    const WeakUseVector &Uses = P.second->second;

    // For complicated expressions like 'a.b.c' and 'x.b.c', WeakObjectProfileTy
    // may not contain enough information to determine that these are different
    // properties. We can only be 100% sure of a repeated use in certain cases,
    // and we adjust the diagnostic kind accordingly so that the less certain
    // case can be turned off if it is too noisy.
    unsigned DiagKind;
    if (Key.isExactProfile())
      DiagKind = diag::warn_arc_repeated_use_of_weak;
    else
      DiagKind = diag::warn_arc_possible_repeated_use_of_weak;

    // Classify the weak object being accessed for better warning text.
    // This enum should stay in sync with the cases in
    // warn_arc_repeated_use_of_weak and warn_arc_possible_repeated_use_of_weak.
    enum {
      Variable,
      Property,
      ImplicitProperty,
      Ivar
    } ObjectKind;

    const NamedDecl *KeyProp = Key.getProperty();
    if (isa<VarDecl>(KeyProp))
      ObjectKind = Variable;
    else if (isa<ObjCPropertyDecl>(KeyProp))
      ObjectKind = Property;
    else if (isa<ObjCMethodDecl>(KeyProp))
      ObjectKind = ImplicitProperty;
    else if (isa<ObjCIvarDecl>(KeyProp))
      ObjectKind = Ivar;
    else
      llvm_unreachable("Unexpected weak object kind!");

    // Do not warn about IBOutlet weak property receivers being set to null
    // since they are typically only used from the main thread.
    if (const ObjCPropertyDecl *Prop = dyn_cast<ObjCPropertyDecl>(KeyProp))
      if (Prop->hasAttr<IBOutletAttr>())
        continue;

    // Show the first time the object was read.
    S.Diag(FirstRead->getBeginLoc(), DiagKind)
        << int(ObjectKind) << KeyProp << int(FunctionKind)
        << FirstRead->getSourceRange();

    // Print all the other accesses as notes.
    for (const auto &Use : Uses) {
      if (Use.getUseExpr() == FirstRead)
        continue;
      S.Diag(Use.getUseExpr()->getBeginLoc(),
             diag::note_arc_weak_also_accessed_here)
          << Use.getUseExpr()->getSourceRange();
    }
  }
}

namespace clang {
namespace {
typedef SmallVector<PartialDiagnosticAt, 1> OptionalNotes;
typedef std::pair<PartialDiagnosticAt, OptionalNotes> DelayedDiag;
typedef std::list<DelayedDiag> DiagList;

struct SortDiagBySourceLocation {
  SourceManager &SM;
  SortDiagBySourceLocation(SourceManager &SM) : SM(SM) {}

  bool operator()(const DelayedDiag &left, const DelayedDiag &right) {
    // Although this call will be slow, this is only called when outputting
    // multiple warnings.
    return SM.isBeforeInTranslationUnit(left.first.first, right.first.first);
  }
};
} // anonymous namespace
} // namespace clang

namespace {
class UninitValsDiagReporter : public UninitVariablesHandler {
  Sema &S;
  typedef SmallVector<UninitUse, 2> UsesVec;
  typedef llvm::PointerIntPair<UsesVec *, 1, bool> MappedType;
  // Prefer using MapVector to DenseMap, so that iteration order will be
  // the same as insertion order. This is needed to obtain a deterministic
  // order of diagnostics when calling flushDiagnostics().
  typedef llvm::MapVector<const VarDecl *, MappedType> UsesMap;
  UsesMap uses;
  UsesMap constRefUses;

public:
  UninitValsDiagReporter(Sema &S) : S(S) {}
  ~UninitValsDiagReporter() override { flushDiagnostics(); }

  MappedType &getUses(UsesMap &um, const VarDecl *vd) {
    MappedType &V = um[vd];
    if (!V.getPointer())
      V.setPointer(new UsesVec());
    return V;
  }

  void handleUseOfUninitVariable(const VarDecl *vd,
                                 const UninitUse &use) override {
    getUses(uses, vd).getPointer()->push_back(use);
  }

  void handleConstRefUseOfUninitVariable(const VarDecl *vd,
                                         const UninitUse &use) override {
    getUses(constRefUses, vd).getPointer()->push_back(use);
  }

  void handleSelfInit(const VarDecl *vd) override {
    getUses(uses, vd).setInt(true);
    getUses(constRefUses, vd).setInt(true);
  }

  void flushDiagnostics() {
    for (const auto &P : uses) {
      const VarDecl *vd = P.first;
      const MappedType &V = P.second;

      UsesVec *vec = V.getPointer();
      bool hasSelfInit = V.getInt();

      // Specially handle the case where we have uses of an uninitialized
      // variable, but the root cause is an idiomatic self-init.  We want
      // to report the diagnostic at the self-init since that is the root cause.
      if (!vec->empty() && hasSelfInit && hasAlwaysUninitializedUse(vec))
        DiagnoseUninitializedUse(S, vd,
                                 UninitUse(vd->getInit()->IgnoreParenCasts(),
                                           /* isAlwaysUninit */ true),
                                 /* alwaysReportSelfInit */ true);
      else {
        // Sort the uses by their SourceLocations.  While not strictly
        // guaranteed to produce them in line/column order, this will provide
        // a stable ordering.
        llvm::sort(*vec, [](const UninitUse &a, const UninitUse &b) {
          // Prefer a more confident report over a less confident one.
          if (a.getKind() != b.getKind())
            return a.getKind() > b.getKind();
          return a.getUser()->getBeginLoc() < b.getUser()->getBeginLoc();
        });

        for (const auto &U : *vec) {
          // If we have self-init, downgrade all uses to 'may be uninitialized'.
          UninitUse Use = hasSelfInit ? UninitUse(U.getUser(), false) : U;

          if (DiagnoseUninitializedUse(S, vd, Use))
            // Skip further diagnostics for this variable. We try to warn only
            // on the first point at which a variable is used uninitialized.
            break;
        }
      }

      // Release the uses vector.
      delete vec;
    }

    uses.clear();

    // Flush all const reference uses diags.
    for (const auto &P : constRefUses) {
      const VarDecl *vd = P.first;
      const MappedType &V = P.second;

      UsesVec *vec = V.getPointer();
      bool hasSelfInit = V.getInt();

      if (!vec->empty() && hasSelfInit && hasAlwaysUninitializedUse(vec))
        DiagnoseUninitializedUse(S, vd,
                                 UninitUse(vd->getInit()->IgnoreParenCasts(),
                                           /* isAlwaysUninit */ true),
                                 /* alwaysReportSelfInit */ true);
      else {
        for (const auto &U : *vec) {
          if (DiagnoseUninitializedConstRefUse(S, vd, U))
            break;
        }
      }

      // Release the uses vector.
      delete vec;
    }

    constRefUses.clear();
  }

private:
  static bool hasAlwaysUninitializedUse(const UsesVec* vec) {
    return llvm::any_of(*vec, [](const UninitUse &U) {
      return U.getKind() == UninitUse::Always ||
             U.getKind() == UninitUse::AfterCall ||
             U.getKind() == UninitUse::AfterDecl;
    });
  }
};

/// Inter-procedural data for the called-once checker.
class CalledOnceInterProceduralData {
public:
  // Add the delayed warning for the given block.
  void addDelayedWarning(const BlockDecl *Block,
                         PartialDiagnosticAt &&Warning) {
    DelayedBlockWarnings[Block].emplace_back(std::move(Warning));
  }
  // Report all of the warnings we've gathered for the given block.
  void flushWarnings(const BlockDecl *Block, Sema &S) {
    for (const PartialDiagnosticAt &Delayed : DelayedBlockWarnings[Block])
      S.Diag(Delayed.first, Delayed.second);

    discardWarnings(Block);
  }
  // Discard all of the warnings we've gathered for the given block.
  void discardWarnings(const BlockDecl *Block) {
    DelayedBlockWarnings.erase(Block);
  }

private:
  using DelayedDiagnostics = SmallVector<PartialDiagnosticAt, 2>;
  llvm::DenseMap<const BlockDecl *, DelayedDiagnostics> DelayedBlockWarnings;
};

class CalledOnceCheckReporter : public CalledOnceCheckHandler {
public:
  CalledOnceCheckReporter(Sema &S, CalledOnceInterProceduralData &Data)
      : S(S), Data(Data) {}
  void handleDoubleCall(const ParmVarDecl *Parameter, const Expr *Call,
                        const Expr *PrevCall, bool IsCompletionHandler,
                        bool Poised) override {
    auto DiagToReport = IsCompletionHandler
                            ? diag::warn_completion_handler_called_twice
                            : diag::warn_called_once_gets_called_twice;
    S.Diag(Call->getBeginLoc(), DiagToReport) << Parameter;
    S.Diag(PrevCall->getBeginLoc(), diag::note_called_once_gets_called_twice)
        << Poised;
  }

  void handleNeverCalled(const ParmVarDecl *Parameter,
                         bool IsCompletionHandler) override {
    auto DiagToReport = IsCompletionHandler
                            ? diag::warn_completion_handler_never_called
                            : diag::warn_called_once_never_called;
    S.Diag(Parameter->getBeginLoc(), DiagToReport)
        << Parameter << /* Captured */ false;
  }

  void handleNeverCalled(const ParmVarDecl *Parameter, const Decl *Function,
                         const Stmt *Where, NeverCalledReason Reason,
                         bool IsCalledDirectly,
                         bool IsCompletionHandler) override {
    auto DiagToReport = IsCompletionHandler
                            ? diag::warn_completion_handler_never_called_when
                            : diag::warn_called_once_never_called_when;
    PartialDiagnosticAt Warning(Where->getBeginLoc(), S.PDiag(DiagToReport)
                                                          << Parameter
                                                          << IsCalledDirectly
                                                          << (unsigned)Reason);

    if (const auto *Block = dyn_cast<BlockDecl>(Function)) {
      // We shouldn't report these warnings on blocks immediately
      Data.addDelayedWarning(Block, std::move(Warning));
    } else {
      S.Diag(Warning.first, Warning.second);
    }
  }

  void handleCapturedNeverCalled(const ParmVarDecl *Parameter,
                                 const Decl *Where,
                                 bool IsCompletionHandler) override {
    auto DiagToReport = IsCompletionHandler
                            ? diag::warn_completion_handler_never_called
                            : diag::warn_called_once_never_called;
    S.Diag(Where->getBeginLoc(), DiagToReport)
        << Parameter << /* Captured */ true;
  }

  void
  handleBlockThatIsGuaranteedToBeCalledOnce(const BlockDecl *Block) override {
    Data.flushWarnings(Block, S);
  }

  void handleBlockWithNoGuarantees(const BlockDecl *Block) override {
    Data.discardWarnings(Block);
  }

private:
  Sema &S;
  CalledOnceInterProceduralData &Data;
};

constexpr unsigned CalledOnceWarnings[] = {
    diag::warn_called_once_never_called,
    diag::warn_called_once_never_called_when,
    diag::warn_called_once_gets_called_twice};

constexpr unsigned CompletionHandlerWarnings[]{
    diag::warn_completion_handler_never_called,
    diag::warn_completion_handler_never_called_when,
    diag::warn_completion_handler_called_twice};

bool shouldAnalyzeCalledOnceImpl(llvm::ArrayRef<unsigned> DiagIDs,
                                 const DiagnosticsEngine &Diags,
                                 SourceLocation At) {
  return llvm::any_of(DiagIDs, [&Diags, At](unsigned DiagID) {
    return !Diags.isIgnored(DiagID, At);
  });
}

bool shouldAnalyzeCalledOnceConventions(const DiagnosticsEngine &Diags,
                                        SourceLocation At) {
  return shouldAnalyzeCalledOnceImpl(CompletionHandlerWarnings, Diags, At);
}

bool shouldAnalyzeCalledOnceParameters(const DiagnosticsEngine &Diags,
                                       SourceLocation At) {
  return shouldAnalyzeCalledOnceImpl(CalledOnceWarnings, Diags, At) ||
         shouldAnalyzeCalledOnceConventions(Diags, At);
}
} // anonymous namespace

//===----------------------------------------------------------------------===//
// -Wthread-safety
//===----------------------------------------------------------------------===//
namespace clang {
namespace threadSafety {
namespace {
class ThreadSafetyReporter : public clang::threadSafety::ThreadSafetyHandler {
  Sema &S;
  DiagList Warnings;
  SourceLocation FunLocation, FunEndLocation;

  const FunctionDecl *CurrentFunction;
  bool Verbose;

  OptionalNotes getNotes() const {
    if (Verbose && CurrentFunction) {
      PartialDiagnosticAt FNote(CurrentFunction->getBody()->getBeginLoc(),
                                S.PDiag(diag::note_thread_warning_in_fun)
                                    << CurrentFunction);
      return OptionalNotes(1, FNote);
    }
    return OptionalNotes();
  }

  OptionalNotes getNotes(const PartialDiagnosticAt &Note) const {
    OptionalNotes ONS(1, Note);
    if (Verbose && CurrentFunction) {
      PartialDiagnosticAt FNote(CurrentFunction->getBody()->getBeginLoc(),
                                S.PDiag(diag::note_thread_warning_in_fun)
                                    << CurrentFunction);
      ONS.push_back(std::move(FNote));
    }
    return ONS;
  }

  OptionalNotes getNotes(const PartialDiagnosticAt &Note1,
                         const PartialDiagnosticAt &Note2) const {
    OptionalNotes ONS;
    ONS.push_back(Note1);
    ONS.push_back(Note2);
    if (Verbose && CurrentFunction) {
      PartialDiagnosticAt FNote(CurrentFunction->getBody()->getBeginLoc(),
                                S.PDiag(diag::note_thread_warning_in_fun)
                                    << CurrentFunction);
      ONS.push_back(std::move(FNote));
    }
    return ONS;
  }

  OptionalNotes makeLockedHereNote(SourceLocation LocLocked, StringRef Kind) {
    return LocLocked.isValid()
               ? getNotes(PartialDiagnosticAt(
                     LocLocked, S.PDiag(diag::note_locked_here) << Kind))
               : getNotes();
  }

  OptionalNotes makeUnlockedHereNote(SourceLocation LocUnlocked,
                                     StringRef Kind) {
    return LocUnlocked.isValid()
               ? getNotes(PartialDiagnosticAt(
                     LocUnlocked, S.PDiag(diag::note_unlocked_here) << Kind))
               : getNotes();
  }

 public:
  ThreadSafetyReporter(Sema &S, SourceLocation FL, SourceLocation FEL)
    : S(S), FunLocation(FL), FunEndLocation(FEL),
      CurrentFunction(nullptr), Verbose(false) {}

  void setVerbose(bool b) { Verbose = b; }

  /// Emit all buffered diagnostics in order of sourcelocation.
  /// We need to output diagnostics produced while iterating through
  /// the lockset in deterministic order, so this function orders diagnostics
  /// and outputs them.
  void emitDiagnostics() {
    Warnings.sort(SortDiagBySourceLocation(S.getSourceManager()));
    for (const auto &Diag : Warnings) {
      S.Diag(Diag.first.first, Diag.first.second);
      for (const auto &Note : Diag.second)
        S.Diag(Note.first, Note.second);
    }
  }

  void handleInvalidLockExp(SourceLocation Loc) override {
    PartialDiagnosticAt Warning(Loc, S.PDiag(diag::warn_cannot_resolve_lock)
                                         << Loc);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleUnmatchedUnlock(StringRef Kind, Name LockName, SourceLocation Loc,
                             SourceLocation LocPreviousUnlock) override {
    if (Loc.isInvalid())
      Loc = FunLocation;
    PartialDiagnosticAt Warning(Loc, S.PDiag(diag::warn_unlock_but_no_lock)
                                         << Kind << LockName);
    Warnings.emplace_back(std::move(Warning),
                          makeUnlockedHereNote(LocPreviousUnlock, Kind));
  }

  void handleIncorrectUnlockKind(StringRef Kind, Name LockName,
                                 LockKind Expected, LockKind Received,
                                 SourceLocation LocLocked,
                                 SourceLocation LocUnlock) override {
    if (LocUnlock.isInvalid())
      LocUnlock = FunLocation;
    PartialDiagnosticAt Warning(
        LocUnlock, S.PDiag(diag::warn_unlock_kind_mismatch)
                       << Kind << LockName << Received << Expected);
    Warnings.emplace_back(std::move(Warning),
                          makeLockedHereNote(LocLocked, Kind));
  }

  void handleDoubleLock(StringRef Kind, Name LockName, SourceLocation LocLocked,
                        SourceLocation LocDoubleLock) override {
    if (LocDoubleLock.isInvalid())
      LocDoubleLock = FunLocation;
    PartialDiagnosticAt Warning(LocDoubleLock, S.PDiag(diag::warn_double_lock)
                                                   << Kind << LockName);
    Warnings.emplace_back(std::move(Warning),
                          makeLockedHereNote(LocLocked, Kind));
  }

  void handleMutexHeldEndOfScope(StringRef Kind, Name LockName,
                                 SourceLocation LocLocked,
                                 SourceLocation LocEndOfScope,
                                 LockErrorKind LEK) override {
    unsigned DiagID = 0;
    switch (LEK) {
      case LEK_LockedSomePredecessors:
        DiagID = diag::warn_lock_some_predecessors;
        break;
      case LEK_LockedSomeLoopIterations:
        DiagID = diag::warn_expecting_lock_held_on_loop;
        break;
      case LEK_LockedAtEndOfFunction:
        DiagID = diag::warn_no_unlock;
        break;
      case LEK_NotLockedAtEndOfFunction:
        DiagID = diag::warn_expecting_locked;
        break;
    }
    if (LocEndOfScope.isInvalid())
      LocEndOfScope = FunEndLocation;

    PartialDiagnosticAt Warning(LocEndOfScope, S.PDiag(DiagID) << Kind
                                                               << LockName);
    Warnings.emplace_back(std::move(Warning),
                          makeLockedHereNote(LocLocked, Kind));
  }

  void handleExclusiveAndShared(StringRef Kind, Name LockName,
                                SourceLocation Loc1,
                                SourceLocation Loc2) override {
    PartialDiagnosticAt Warning(Loc1,
                                S.PDiag(diag::warn_lock_exclusive_and_shared)
                                    << Kind << LockName);
    PartialDiagnosticAt Note(Loc2, S.PDiag(diag::note_lock_exclusive_and_shared)
                                       << Kind << LockName);
    Warnings.emplace_back(std::move(Warning), getNotes(Note));
  }

  void handleNoMutexHeld(const NamedDecl *D, ProtectedOperationKind POK,
                         AccessKind AK, SourceLocation Loc) override {
    assert((POK == POK_VarAccess || POK == POK_VarDereference) &&
           "Only works for variables");
    unsigned DiagID = POK == POK_VarAccess?
                        diag::warn_variable_requires_any_lock:
                        diag::warn_var_deref_requires_any_lock;
    PartialDiagnosticAt Warning(Loc, S.PDiag(DiagID)
      << D << getLockKindFromAccessKind(AK));
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleMutexNotHeld(StringRef Kind, const NamedDecl *D,
                          ProtectedOperationKind POK, Name LockName,
                          LockKind LK, SourceLocation Loc,
                          Name *PossibleMatch) override {
    unsigned DiagID = 0;
    if (PossibleMatch) {
      switch (POK) {
        case POK_VarAccess:
          DiagID = diag::warn_variable_requires_lock_precise;
          break;
        case POK_VarDereference:
          DiagID = diag::warn_var_deref_requires_lock_precise;
          break;
        case POK_FunctionCall:
          DiagID = diag::warn_fun_requires_lock_precise;
          break;
        case POK_PassByRef:
          DiagID = diag::warn_guarded_pass_by_reference;
          break;
        case POK_PtPassByRef:
          DiagID = diag::warn_pt_guarded_pass_by_reference;
          break;
        case POK_ReturnByRef:
          DiagID = diag::warn_guarded_return_by_reference;
          break;
        case POK_PtReturnByRef:
          DiagID = diag::warn_pt_guarded_return_by_reference;
          break;
      }
      PartialDiagnosticAt Warning(Loc, S.PDiag(DiagID) << Kind
                                                       << D
                                                       << LockName << LK);
      PartialDiagnosticAt Note(Loc, S.PDiag(diag::note_found_mutex_near_match)
                                        << *PossibleMatch);
      if (Verbose && POK == POK_VarAccess) {
        PartialDiagnosticAt VNote(D->getLocation(),
                                  S.PDiag(diag::note_guarded_by_declared_here)
                                      << D->getDeclName());
        Warnings.emplace_back(std::move(Warning), getNotes(Note, VNote));
      } else
        Warnings.emplace_back(std::move(Warning), getNotes(Note));
    } else {
      switch (POK) {
        case POK_VarAccess:
          DiagID = diag::warn_variable_requires_lock;
          break;
        case POK_VarDereference:
          DiagID = diag::warn_var_deref_requires_lock;
          break;
        case POK_FunctionCall:
          DiagID = diag::warn_fun_requires_lock;
          break;
        case POK_PassByRef:
          DiagID = diag::warn_guarded_pass_by_reference;
          break;
        case POK_PtPassByRef:
          DiagID = diag::warn_pt_guarded_pass_by_reference;
          break;
        case POK_ReturnByRef:
          DiagID = diag::warn_guarded_return_by_reference;
          break;
        case POK_PtReturnByRef:
          DiagID = diag::warn_pt_guarded_return_by_reference;
          break;
      }
      PartialDiagnosticAt Warning(Loc, S.PDiag(DiagID) << Kind
                                                       << D
                                                       << LockName << LK);
      if (Verbose && POK == POK_VarAccess) {
        PartialDiagnosticAt Note(D->getLocation(),
                                 S.PDiag(diag::note_guarded_by_declared_here));
        Warnings.emplace_back(std::move(Warning), getNotes(Note));
      } else
        Warnings.emplace_back(std::move(Warning), getNotes());
    }
  }

  void handleNegativeNotHeld(StringRef Kind, Name LockName, Name Neg,
                             SourceLocation Loc) override {
    PartialDiagnosticAt Warning(Loc,
        S.PDiag(diag::warn_acquire_requires_negative_cap)
        << Kind << LockName << Neg);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleNegativeNotHeld(const NamedDecl *D, Name LockName,
                             SourceLocation Loc) override {
    PartialDiagnosticAt Warning(
        Loc, S.PDiag(diag::warn_fun_requires_negative_cap) << D << LockName);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleFunExcludesLock(StringRef Kind, Name FunName, Name LockName,
                             SourceLocation Loc) override {
    PartialDiagnosticAt Warning(Loc, S.PDiag(diag::warn_fun_excludes_mutex)
                                         << Kind << FunName << LockName);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleLockAcquiredBefore(StringRef Kind, Name L1Name, Name L2Name,
                                SourceLocation Loc) override {
    PartialDiagnosticAt Warning(Loc,
      S.PDiag(diag::warn_acquired_before) << Kind << L1Name << L2Name);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void handleBeforeAfterCycle(Name L1Name, SourceLocation Loc) override {
    PartialDiagnosticAt Warning(Loc,
      S.PDiag(diag::warn_acquired_before_after_cycle) << L1Name);
    Warnings.emplace_back(std::move(Warning), getNotes());
  }

  void enterFunction(const FunctionDecl* FD) override {
    CurrentFunction = FD;
  }

  void leaveFunction(const FunctionDecl* FD) override {
    CurrentFunction = nullptr;
  }
};
} // anonymous namespace
} // namespace threadSafety
} // namespace clang

//===----------------------------------------------------------------------===//
// -Wconsumed
//===----------------------------------------------------------------------===//

namespace clang {
namespace consumed {
namespace {
class ConsumedWarningsHandler : public ConsumedWarningsHandlerBase {

  Sema &S;
  DiagList Warnings;

public:

  ConsumedWarningsHandler(Sema &S) : S(S) {}

  void emitDiagnostics() override {
    Warnings.sort(SortDiagBySourceLocation(S.getSourceManager()));
    for (const auto &Diag : Warnings) {
      S.Diag(Diag.first.first, Diag.first.second);
      for (const auto &Note : Diag.second)
        S.Diag(Note.first, Note.second);
    }
  }

  void warnLoopStateMismatch(SourceLocation Loc,
                             StringRef VariableName) override {
    PartialDiagnosticAt Warning(Loc, S.PDiag(diag::warn_loop_state_mismatch) <<
      VariableName);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnParamReturnTypestateMismatch(SourceLocation Loc,
                                        StringRef VariableName,
                                        StringRef ExpectedState,
                                        StringRef ObservedState) override {

    PartialDiagnosticAt Warning(Loc, S.PDiag(
      diag::warn_param_return_typestate_mismatch) << VariableName <<
        ExpectedState << ObservedState);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnParamTypestateMismatch(SourceLocation Loc, StringRef ExpectedState,
                                  StringRef ObservedState) override {

    PartialDiagnosticAt Warning(Loc, S.PDiag(
      diag::warn_param_typestate_mismatch) << ExpectedState << ObservedState);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnReturnTypestateForUnconsumableType(SourceLocation Loc,
                                              StringRef TypeName) override {
    PartialDiagnosticAt Warning(Loc, S.PDiag(
      diag::warn_return_typestate_for_unconsumable_type) << TypeName);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnReturnTypestateMismatch(SourceLocation Loc, StringRef ExpectedState,
                                   StringRef ObservedState) override {

    PartialDiagnosticAt Warning(Loc, S.PDiag(
      diag::warn_return_typestate_mismatch) << ExpectedState << ObservedState);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnUseOfTempInInvalidState(StringRef MethodName, StringRef State,
                                   SourceLocation Loc) override {

    PartialDiagnosticAt Warning(Loc, S.PDiag(
      diag::warn_use_of_temp_in_invalid_state) << MethodName << State);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }

  void warnUseInInvalidState(StringRef MethodName, StringRef VariableName,
                             StringRef State, SourceLocation Loc) override {

    PartialDiagnosticAt Warning(Loc, S.PDiag(diag::warn_use_in_invalid_state) <<
                                MethodName << VariableName << State);

    Warnings.emplace_back(std::move(Warning), OptionalNotes());
  }
};
} // anonymous namespace
} // namespace consumed
} // namespace clang

//===----------------------------------------------------------------------===//
// Unsafe buffer usage analysis.
//===----------------------------------------------------------------------===//

namespace {
class UnsafeBufferUsageReporter : public UnsafeBufferUsageHandler {
  Sema &S;
  bool SuggestSuggestions;  // Recommend -fsafe-buffer-usage-suggestions?

  // Lists as a string the names of variables in `VarGroupForVD` except for `VD`
  // itself:
  std::string listVariableGroupAsString(
      const VarDecl *VD, const ArrayRef<const VarDecl *> &VarGroupForVD) const {
    if (VarGroupForVD.size() <= 1)
      return "";

    std::vector<StringRef> VarNames;
    auto PutInQuotes = [](StringRef S) -> std::string {
      return "'" + S.str() + "'";
    };

    for (auto *V : VarGroupForVD) {
      if (V == VD)
        continue;
      VarNames.push_back(V->getName());
    }
    if (VarNames.size() == 1) {
      return PutInQuotes(VarNames[0]);
    }
    if (VarNames.size() == 2) {
      return PutInQuotes(VarNames[0]) + " and " + PutInQuotes(VarNames[1]);
    }
    assert(VarGroupForVD.size() > 3);
    const unsigned N = VarNames.size() -
                       2; // need to print the last two names as "..., X, and Y"
    std::string AllVars = "";

    for (unsigned I = 0; I < N; ++I)
      AllVars.append(PutInQuotes(VarNames[I]) + ", ");
    AllVars.append(PutInQuotes(VarNames[N]) + ", and " +
                   PutInQuotes(VarNames[N + 1]));
    return AllVars;
  }

public:
  UnsafeBufferUsageReporter(Sema &S, bool SuggestSuggestions)
    : S(S), SuggestSuggestions(SuggestSuggestions) {}

  void handleUnsafeOperation(const Stmt *Operation, bool IsRelatedToDecl,
                             ASTContext &Ctx) override {
    SourceLocation Loc;
    SourceRange Range;
    unsigned MsgParam = 0;
    if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(Operation)) {
      Loc = ASE->getBase()->getExprLoc();
      Range = ASE->getBase()->getSourceRange();
      MsgParam = 2;
    } else if (const auto *BO = dyn_cast<BinaryOperator>(Operation)) {
      BinaryOperator::Opcode Op = BO->getOpcode();
      if (Op == BO_Add || Op == BO_AddAssign || Op == BO_Sub ||
          Op == BO_SubAssign) {
        if (BO->getRHS()->getType()->isIntegerType()) {
          Loc = BO->getLHS()->getExprLoc();
          Range = BO->getLHS()->getSourceRange();
        } else {
          Loc = BO->getRHS()->getExprLoc();
          Range = BO->getRHS()->getSourceRange();
        }
        MsgParam = 1;
      }
    } else if (const auto *UO = dyn_cast<UnaryOperator>(Operation)) {
      UnaryOperator::Opcode Op = UO->getOpcode();
      if (Op == UO_PreInc || Op == UO_PreDec || Op == UO_PostInc ||
          Op == UO_PostDec) {
        Loc = UO->getSubExpr()->getExprLoc();
        Range = UO->getSubExpr()->getSourceRange();
        MsgParam = 1;
      }
    } else {
      if (isa<CallExpr>(Operation) || isa<CXXConstructExpr>(Operation)) {
        // note_unsafe_buffer_operation doesn't have this mode yet.
        assert(!IsRelatedToDecl && "Not implemented yet!");
        MsgParam = 3;
      } else if (const auto *ECE = dyn_cast<ExplicitCastExpr>(Operation)) {
        QualType destType = ECE->getType();
        if (!isa<PointerType>(destType))
          return;

        const uint64_t dSize =
            Ctx.getTypeSize(destType.getTypePtr()->getPointeeType());

        QualType srcType = ECE->getSubExpr()->getType();
        const uint64_t sSize =
            Ctx.getTypeSize(srcType.getTypePtr()->getPointeeType());
        if (sSize >= dSize)
          return;

        MsgParam = 4;
      }
      Loc = Operation->getBeginLoc();
      Range = Operation->getSourceRange();
    }
    if (IsRelatedToDecl) {
      assert(!SuggestSuggestions &&
             "Variables blamed for unsafe buffer usage without suggestions!");
      S.Diag(Loc, diag::note_unsafe_buffer_operation) << MsgParam << Range;
    } else {
      S.Diag(Loc, diag::warn_unsafe_buffer_operation) << MsgParam << Range;
      if (SuggestSuggestions) {
        S.Diag(Loc, diag::note_safe_buffer_usage_suggestions_disabled);
      }
    }
  }

  void handleUnsafeOperationInContainer(const Stmt *Operation,
                                        bool IsRelatedToDecl,
                                        ASTContext &Ctx) override {
    SourceLocation Loc;
    SourceRange Range;
    unsigned MsgParam = 0;

    // This function only handles SpanTwoParamConstructorGadget so far, which
    // always gives a CXXConstructExpr.
    const auto *CtorExpr = cast<CXXConstructExpr>(Operation);
    Loc = CtorExpr->getLocation();

    S.Diag(Loc, diag::warn_unsafe_buffer_usage_in_container);
    if (IsRelatedToDecl) {
      assert(!SuggestSuggestions &&
             "Variables blamed for unsafe buffer usage without suggestions!");
      S.Diag(Loc, diag::note_unsafe_buffer_operation) << MsgParam << Range;
    }
  }

  void handleUnsafeVariableGroup(const VarDecl *Variable,
                                 const VariableGroupsManager &VarGrpMgr,
                                 FixItList &&Fixes, const Decl *D,
                                 const FixitStrategy &VarTargetTypes) override {
    assert(!SuggestSuggestions &&
           "Unsafe buffer usage fixits displayed without suggestions!");
    S.Diag(Variable->getLocation(), diag::warn_unsafe_buffer_variable)
        << Variable << (Variable->getType()->isPointerType() ? 0 : 1)
        << Variable->getSourceRange();
    if (!Fixes.empty()) {
      assert(isa<NamedDecl>(D) &&
             "Fix-its are generated only for `NamedDecl`s");
      const NamedDecl *ND = cast<NamedDecl>(D);
      bool BriefMsg = false;
      // If the variable group involves parameters, the diagnostic message will
      // NOT explain how the variables are grouped as the reason is non-trivial
      // and irrelavant to users' experience:
      const auto VarGroupForVD = VarGrpMgr.getGroupOfVar(Variable, &BriefMsg);
      unsigned FixItStrategy = 0;
      switch (VarTargetTypes.lookup(Variable)) {
      case clang::FixitStrategy::Kind::Span:
        FixItStrategy = 0;
        break;
      case clang::FixitStrategy::Kind::Array:
        FixItStrategy = 1;
        break;
      default:
        assert(false && "We support only std::span and std::array");
      };

      const auto &FD =
          S.Diag(Variable->getLocation(),
                 BriefMsg ? diag::note_unsafe_buffer_variable_fixit_together
                          : diag::note_unsafe_buffer_variable_fixit_group);

      FD << Variable << FixItStrategy;
      FD << listVariableGroupAsString(Variable, VarGroupForVD)
         << (VarGroupForVD.size() > 1) << ND;
      for (const auto &F : Fixes) {
        FD << F;
      }
    }

#ifndef NDEBUG
    if (areDebugNotesRequested())
      for (const DebugNote &Note: DebugNotesByVar[Variable])
        S.Diag(Note.first, diag::note_safe_buffer_debug_mode) << Note.second;
#endif
  }

  bool isSafeBufferOptOut(const SourceLocation &Loc) const override {
    return S.PP.isSafeBufferOptOut(S.getSourceManager(), Loc);
  }

  bool ignoreUnsafeBufferInContainer(const SourceLocation &Loc) const override {
    return S.Diags.isIgnored(diag::warn_unsafe_buffer_usage_in_container, Loc);
  }

  // Returns the text representation of clang::unsafe_buffer_usage attribute.
  // `WSSuffix` holds customized "white-space"s, e.g., newline or whilespace
  // characters.
  std::string
  getUnsafeBufferUsageAttributeTextAt(SourceLocation Loc,
                                      StringRef WSSuffix = "") const override {
    Preprocessor &PP = S.getPreprocessor();
    TokenValue ClangUnsafeBufferUsageTokens[] = {
        tok::l_square,
        tok::l_square,
        PP.getIdentifierInfo("clang"),
        tok::coloncolon,
        PP.getIdentifierInfo("unsafe_buffer_usage"),
        tok::r_square,
        tok::r_square};

    StringRef MacroName;

    // The returned macro (it returns) is guaranteed not to be function-like:
    MacroName = PP.getLastMacroWithSpelling(Loc, ClangUnsafeBufferUsageTokens);
    if (MacroName.empty())
      MacroName = "[[clang::unsafe_buffer_usage]]";
    return MacroName.str() + WSSuffix.str();
  }
};
} // namespace

// =============================================================================

namespace FXAnalysis {

enum class DiagnosticID : uint8_t {
  None = 0, // sentinel for an empty Diagnostic
  Throws,
  Catches,
  CallsObjC,
  AllocatesMemory,
  HasStaticLocal,
  AccessesThreadLocal,

  // These only apply to callees, where the analysis stops at the Decl
  DeclDisallowsInference,

  CallsDeclWithoutEffect,
  CallsExprWithoutEffect,
};

// Holds an effect diagnosis, potentially for the entire duration of the
// analysis phase, in order to refer to it when explaining why a caller has been
// made unsafe by a callee.
struct Diagnostic {
  FunctionEffect Effect;
  DiagnosticID ID = DiagnosticID::None;
  SourceLocation Loc;
  const Decl *Callee = nullptr; // only valid for Calls*

  Diagnostic() = default;

  Diagnostic(const FunctionEffect &Effect, DiagnosticID ID, SourceLocation Loc,
             const Decl *Callee = nullptr)
      : Effect(Effect), ID(ID), Loc(Loc), Callee(Callee) {}
};

enum class SpecialFuncType : uint8_t { None, OperatorNew, OperatorDelete };
enum class CallType {
  // unknown: probably function pointer
  Unknown,
  Function,
  Virtual,
  Block
};

// Return whether a function's effects CAN be verified.
// The question of whether it SHOULD be verified is independent.
static bool functionIsVerifiable(const FunctionDecl *FD) {
  if (!(FD->hasBody() || FD->isInlined())) {
    // externally defined; we couldn't verify if we wanted to.
    return false;
  }
  if (FD->isTrivial()) {
    // Otherwise `struct x { int a; };` would have an unverifiable default
    // constructor.
    return true;
  }
  return true;
}

/// A mutable set of FunctionEffect, for use in places where any conditions
/// have been resolved or can be ignored.
class EffectSet {
  // This implementation optimizes footprint, since we hold one of these for
  // every function visited, which, due to inference, can be many more functions
  // than have declared effects.

  template <typename T, typename SizeT, SizeT Capacity> struct FixedVector {
    SizeT Count = 0;
    T Items[Capacity] = {};

    using value_type = T;

    using iterator = T *;
    using const_iterator = const T *;
    iterator begin() { return &Items[0]; }
    iterator end() { return &Items[Count]; }
    const_iterator begin() const { return &Items[0]; }
    const_iterator end() const { return &Items[Count]; }
    const_iterator cbegin() const { return &Items[0]; }
    const_iterator cend() const { return &Items[Count]; }

    void insert(iterator I, const T &Value) {
      assert(Count < Capacity);
      iterator E = end();
      if (I != E)
        std::copy_backward(I, E, E + 1);
      *I = Value;
      ++Count;
    }

    void push_back(const T &Value) {
      assert(Count < Capacity);
      Items[Count++] = Value;
    }
  };

  // As long as FunctionEffect is only 1 byte, and there are only 2 verifiable
  // effects, this fixed-size vector with a capacity of 7 is more than
  // sufficient and is only 8 bytes.
  FixedVector<FunctionEffect, uint8_t, 7> Impl;

public:
  EffectSet() = default;
  explicit EffectSet(FunctionEffectsRef FX) { insert(FX); }

  operator ArrayRef<FunctionEffect>() const {
    return ArrayRef(Impl.cbegin(), Impl.cend());
  }

  using iterator = const FunctionEffect *;
  iterator begin() const { return Impl.cbegin(); }
  iterator end() const { return Impl.cend(); }

  void insert(const FunctionEffect &Effect) {
    FunctionEffect *Iter = Impl.begin();
    FunctionEffect *End = Impl.end();
    // linear search; lower_bound is overkill for a tiny vector like this
    for (; Iter != End; ++Iter) {
      if (*Iter == Effect)
        return;
      if (Effect < *Iter)
        break;
    }
    Impl.insert(Iter, Effect);
  }
  void insert(const EffectSet &Set) {
    for (const FunctionEffect &Item : Set) {
      // push_back because set is already sorted
      Impl.push_back(Item);
    }
  }
  void insert(FunctionEffectsRef FX) {
    for (const FunctionEffectWithCondition &EC : FX) {
      assert(EC.Cond.getCondition() ==
             nullptr); // should be resolved by now, right?
      // push_back because set is already sorted
      Impl.push_back(EC.Effect);
    }
  }
  bool contains(const FunctionEffect::Kind EK) const {
    for (const FunctionEffect &E : Impl)
      if (E.kind() == EK)
        return true;
    return false;
  }

  void dump(llvm::raw_ostream &OS) const;

  static EffectSet difference(ArrayRef<FunctionEffect> LHS,
                              ArrayRef<FunctionEffect> RHS) {
    EffectSet Result;
    std::set_difference(LHS.begin(), LHS.end(), RHS.begin(), RHS.end(),
                        std::back_inserter(Result.Impl));
    return Result;
  }
};

LLVM_DUMP_METHOD void EffectSet::dump(llvm::raw_ostream &OS) const {
  OS << "Effects{";
  bool First = true;
  for (const FunctionEffect &Effect : *this) {
    if (!First)
      OS << ", ";
    else
      First = false;
    OS << Effect.name();
  }
  OS << "}";
}

// Transitory, more extended information about a callable, which can be a
// function, block, function pointer, etc.
struct CallableInfo {
  // CDecl holds the function's definition, if any.
  // FunctionDecl if CallType::Function or Virtual
  // BlockDecl if CallType::Block
  const Decl *CDecl;
  mutable std::optional<std::string> MaybeName;
  SpecialFuncType FuncType = SpecialFuncType::None;
  EffectSet Effects;
  CallType CType = CallType::Unknown;

  CallableInfo(Sema &SemaRef, const Decl &CD,
               SpecialFuncType FT = SpecialFuncType::None)
      : CDecl(&CD), FuncType(FT) {
    FunctionEffectsRef FXRef;

    if (auto *FD = dyn_cast<FunctionDecl>(CDecl)) {
      // Use the function's definition, if any.
      if (const FunctionDecl *Def = FD->getDefinition())
        CDecl = FD = Def;
      CType = CallType::Function;
      if (auto *Method = dyn_cast<CXXMethodDecl>(FD);
          Method && Method->isVirtual())
        CType = CallType::Virtual;
      FXRef = FD->getFunctionEffects();
    } else if (auto *BD = dyn_cast<BlockDecl>(CDecl)) {
      CType = CallType::Block;
      FXRef = BD->getFunctionEffects();
    } else if (auto *VD = dyn_cast<ValueDecl>(CDecl)) {
      // ValueDecl is function, enum, or variable, so just look at its type.
      FXRef = FunctionEffectsRef::get(VD->getType());
    }
    Effects = EffectSet(FXRef);
  }

  bool isDirectCall() const {
    return CType == CallType::Function || CType == CallType::Block;
  }

  bool isVerifiable() const {
    switch (CType) {
    case CallType::Unknown:
    case CallType::Virtual:
      break;
    case CallType::Block:
      return true;
    case CallType::Function:
      return functionIsVerifiable(dyn_cast<FunctionDecl>(CDecl));
    }
    return false;
  }

  /// Generate a name for logging and diagnostics.
  std::string name(Sema &Sem) const {
    if (!MaybeName) {
      std::string Name;
      llvm::raw_string_ostream OS(Name);

      if (auto *FD = dyn_cast<FunctionDecl>(CDecl))
        FD->getNameForDiagnostic(OS, Sem.getPrintingPolicy(),
                                 /*Qualified=*/true);
      else if (auto *BD = dyn_cast<BlockDecl>(CDecl))
        OS << "(block " << BD->getBlockManglingNumber() << ")";
      else if (auto *VD = dyn_cast<NamedDecl>(CDecl))
        VD->printQualifiedName(OS);
      MaybeName = Name;
    }
    return *MaybeName;
  }
};

// ----------
// Map effects to single diagnostics, to hold the first (of potentially many)
// diagnostics pertaining to an effect, per function.
class EffectToDiagnosticMap {
  // Since we currently only have a tiny number of effects (typically no more
  // than 1), use a sorted SmallVector with an inline capacity of 1. Since it
  // is often empty, use a unique_ptr to the SmallVector.
  // Note that Diagnostic itself contains a FunctionEffect which is the key.
  using ImplVec = llvm::SmallVector<Diagnostic, 1>;
  std::unique_ptr<ImplVec> Impl;

public:
  // Insert a new diagnostic if we do not already have one for its effect.
  void maybeInsert(const Diagnostic &Diag) {
    if (Impl == nullptr)
      Impl = std::make_unique<ImplVec>();
    auto *Iter = _find(Diag.Effect);
    if (Iter != Impl->end() && Iter->Effect == Diag.Effect)
      return;

    Impl->insert(Iter, Diag);
  }

  const Diagnostic *lookup(FunctionEffect Key) {
    if (Impl == nullptr)
      return nullptr;

    auto *Iter = _find(Key);
    if (Iter != Impl->end() && Iter->Effect == Key)
      return &*Iter;

    return nullptr;
  }

  size_t size() const { return Impl ? Impl->size() : 0; }

private:
  ImplVec::iterator _find(const FunctionEffect &key) {
    // A linear search suffices for a tiny number of possible effects.
    auto *End = Impl->end();
    for (auto *Iter = Impl->begin(); Iter != End; ++Iter)
      if (!(Iter->Effect < key))
        return Iter;
    return End;
  }
};

// ----------
// State pertaining to a function whose AST is walked and whose effect analysis
// is dependent on a subsequent analysis of other functions.
class PendingFunctionAnalysis {
  friend class CompleteFunctionAnalysis;

public:
  struct DirectCall {
    const Decl *Callee;
    SourceLocation CallLoc;
    // Not all recursive calls are detected, just enough
    // to break cycles.
    bool Recursed = false;

    DirectCall(const Decl *D, SourceLocation CallLoc)
        : Callee(D), CallLoc(CallLoc) {}
  };

  // We always have two disjoint sets of effects to verify:
  // 1. Effects declared explicitly by this function.
  // 2. All other inferrable effects needing verification.
  EffectSet DeclaredVerifiableEffects;
  EffectSet FXToInfer;

private:
  // Diagnostics pertaining to the function's explicit effects.
  SmallVector<Diagnostic, 0> DiagnosticsForExplicitFX;

  // Diagnostics pertaining to other, non-explicit, inferrable effects.
  EffectToDiagnosticMap InferrableEffectToFirstDiagnostic;

  // These unverified direct calls are what keeps the analysis "pending",
  // until the callees can be verified.
  SmallVector<DirectCall, 0> UnverifiedDirectCalls;

public:
  PendingFunctionAnalysis(
      Sema &Sem, const CallableInfo &CInfo,
      ArrayRef<FunctionEffect> AllInferrableEffectsToVerify) {
    DeclaredVerifiableEffects = CInfo.Effects;

    // Check for effects we are not allowed to infer
    EffectSet InferrableFX;

    for (const FunctionEffect &effect : AllInferrableEffectsToVerify) {
      if (effect.canInferOnFunction(*CInfo.CDecl))
        InferrableFX.insert(effect);
      else {
        // Add a diagnostic for this effect if a caller were to
        // try to infer it.
        InferrableEffectToFirstDiagnostic.maybeInsert(
            Diagnostic(effect, DiagnosticID::DeclDisallowsInference,
                       CInfo.CDecl->getLocation()));
      }
    }
    // InferrableFX is now the set of inferrable effects which are not
    // prohibited
    FXToInfer = EffectSet::difference(InferrableFX, DeclaredVerifiableEffects);
  }

  // Hide the way that diagnostics for explicitly required effects vs. inferred
  // ones are handled differently.
  void checkAddDiagnostic(bool Inferring, const Diagnostic &NewDiag) {
    if (!Inferring)
      DiagnosticsForExplicitFX.push_back(NewDiag);
    else
      InferrableEffectToFirstDiagnostic.maybeInsert(NewDiag);
  }

  void addUnverifiedDirectCall(const Decl *D, SourceLocation CallLoc) {
    UnverifiedDirectCalls.emplace_back(D, CallLoc);
  }

  // Analysis is complete when there are no unverified direct calls.
  bool isComplete() const { return UnverifiedDirectCalls.empty(); }

  const Diagnostic *diagnosticForInferrableEffect(FunctionEffect effect) {
    return InferrableEffectToFirstDiagnostic.lookup(effect);
  }

  SmallVector<DirectCall, 0> &unverifiedCalls() {
    assert(!isComplete());
    return UnverifiedDirectCalls;
  }

  SmallVector<Diagnostic, 0> &getDiagnosticsForExplicitFX() {
    return DiagnosticsForExplicitFX;
  }

  void dump(Sema &SemaRef, llvm::raw_ostream &OS) const {
    OS << "Pending: Declared ";
    DeclaredVerifiableEffects.dump(OS);
    OS << ", " << DiagnosticsForExplicitFX.size() << " diags; ";
    OS << " Infer ";
    FXToInfer.dump(OS);
    OS << ", " << InferrableEffectToFirstDiagnostic.size() << " diags";
    if (!UnverifiedDirectCalls.empty()) {
      OS << "; Calls: ";
      for (const DirectCall &Call : UnverifiedDirectCalls) {
        CallableInfo CI(SemaRef, *Call.Callee);
        OS << " " << CI.name(SemaRef);
      }
    }
    OS << "\n";
  }
};

// ----------
class CompleteFunctionAnalysis {
  // Current size: 2 pointers
public:
  // Has effects which are both the declared ones -- not to be inferred -- plus
  // ones which have been successfully inferred. These are all considered
  // "verified" for the purposes of callers; any issue with verifying declared
  // effects has already been reported and is not the problem of any caller.
  EffectSet VerifiedEffects;

private:
  // This is used to generate notes about failed inference.
  EffectToDiagnosticMap InferrableEffectToFirstDiagnostic;

public:
  // The incoming Pending analysis is consumed (member(s) are moved-from).
  CompleteFunctionAnalysis(
      ASTContext &Ctx, PendingFunctionAnalysis &Pending,
      const EffectSet &DeclaredEffects,
      ArrayRef<FunctionEffect> AllInferrableEffectsToVerify) {
    VerifiedEffects.insert(DeclaredEffects);
    for (const FunctionEffect &effect : AllInferrableEffectsToVerify)
      if (Pending.diagnosticForInferrableEffect(effect) == nullptr)
        VerifiedEffects.insert(effect);

    InferrableEffectToFirstDiagnostic =
        std::move(Pending.InferrableEffectToFirstDiagnostic);
  }

  const Diagnostic *firstDiagnosticForEffect(const FunctionEffect &Effect) {
    return InferrableEffectToFirstDiagnostic.lookup(Effect);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "Complete: Verified ";
    VerifiedEffects.dump(OS);
    OS << "; Infer ";
    OS << InferrableEffectToFirstDiagnostic.size() << " diags\n";
  }
};

const Decl *CanonicalFunctionDecl(const Decl *D) {
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    FD = FD->getCanonicalDecl();
    assert(FD != nullptr);
    return FD;
  }
  return D;
}

// ==========
class Analyzer {
  constexpr static int DebugLogLevel = 0;
  // --
  Sema &Sem;

  // Subset of Sema.AllEffectsToVerify
  EffectSet AllInferrableEffectsToVerify;

  using FuncAnalysisPtr =
      llvm::PointerUnion<PendingFunctionAnalysis *, CompleteFunctionAnalysis *>;

  // Map all Decls analyzed to FuncAnalysisPtr. Pending state is larger
  // than complete state, so use different objects to represent them.
  // The state pointers are owned by the container.
  class AnalysisMap : protected llvm::DenseMap<const Decl *, FuncAnalysisPtr> {
    using Base = llvm::DenseMap<const Decl *, FuncAnalysisPtr>;

  public:
    ~AnalysisMap();

    // Use non-public inheritance in order to maintain the invariant
    // that lookups and insertions are via the canonical Decls.

    FuncAnalysisPtr lookup(const Decl *Key) const {
      return Base::lookup(CanonicalFunctionDecl(Key));
    }

    FuncAnalysisPtr &operator[](const Decl *Key) {
      return Base::operator[](CanonicalFunctionDecl(Key));
    }

    /// Shortcut for the case where we only care about completed analysis.
    CompleteFunctionAnalysis *completedAnalysisForDecl(const Decl *D) const {
      if (FuncAnalysisPtr AP = lookup(D);
          isa_and_nonnull<CompleteFunctionAnalysis *>(AP))
        return AP.get<CompleteFunctionAnalysis *>();
      return nullptr;
    }

    void dump(Sema &SemaRef, llvm::raw_ostream &OS) {
      OS << "\nAnalysisMap:\n";
      for (const auto &item : *this) {
        CallableInfo CI(SemaRef, *item.first);
        const auto AP = item.second;
        OS << item.first << " " << CI.name(SemaRef) << " : ";
        if (AP.isNull())
          OS << "null\n";
        else if (isa<CompleteFunctionAnalysis *>(AP)) {
          auto *CFA = AP.get<CompleteFunctionAnalysis *>();
          OS << CFA << " ";
          CFA->dump(OS);
        } else if (isa<PendingFunctionAnalysis *>(AP)) {
          auto *PFA = AP.get<PendingFunctionAnalysis *>();
          OS << PFA << " ";
          PFA->dump(SemaRef, OS);
        } else
          llvm_unreachable("never");
      }
      OS << "---\n";
    }
  };
  AnalysisMap DeclAnalysis;

public:
  Analyzer(Sema &S) : Sem(S) {}

  void run(const TranslationUnitDecl &TU) {
    // Gather all of the effects to be verified to see what operations need to
    // be checked, and to see which ones are inferrable.
    for (const FunctionEffectWithCondition &CFE : Sem.AllEffectsToVerify) {
      const FunctionEffect &Effect = CFE.Effect;
      const FunctionEffect::Flags Flags = Effect.flags();
      if (Flags & FunctionEffect::FE_InferrableOnCallees)
        AllInferrableEffectsToVerify.insert(Effect);
    }
    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "AllInferrableEffectsToVerify: ";
      AllInferrableEffectsToVerify.dump(llvm::outs());
      llvm::outs() << "\n";
    }

    // We can use DeclsWithEffectsToVerify as a stack for a
    // depth-first traversal; there's no need for a second container. But first,
    // reverse it, so when working from the end, Decls are verified in the order
    // they are declared.
    SmallVector<const Decl *> &VerificationQueue = Sem.DeclsWithEffectsToVerify;
    std::reverse(VerificationQueue.begin(), VerificationQueue.end());

    while (!VerificationQueue.empty()) {
      const Decl *D = VerificationQueue.back();
      if (FuncAnalysisPtr AP = DeclAnalysis.lookup(D)) {
        if (isa<CompleteFunctionAnalysis *>(AP)) {
          // already done
          VerificationQueue.pop_back();
          continue;
        }
        if (isa<PendingFunctionAnalysis *>(AP)) {
          // All children have been traversed; finish analysis.
          auto *Pending = AP.get<PendingFunctionAnalysis *>();
          finishPendingAnalysis(D, Pending);
          VerificationQueue.pop_back();
          continue;
        }
        llvm_unreachable("unexpected DeclAnalysis item");
      }

      // Not previously visited; begin a new analysis for this Decl.
      PendingFunctionAnalysis *Pending = verifyDecl(D);
      if (Pending == nullptr) {
        // completed now
        VerificationQueue.pop_back();
        continue;
      }

      // Analysis remains pending because there are direct callees to be
      // verified first. Push them onto the queue.
      for (PendingFunctionAnalysis::DirectCall &Call :
           Pending->unverifiedCalls()) {
        FuncAnalysisPtr AP = DeclAnalysis.lookup(Call.Callee);
        if (AP.isNull()) {
          VerificationQueue.push_back(Call.Callee);
          continue;
        }
        if (isa<PendingFunctionAnalysis *>(AP)) {
          // This indicates recursion (not necessarily direct). For the
          // purposes of effect analysis, we can just ignore it since
          // no effects forbid recursion.
          Call.Recursed = true;
          continue;
        }
        llvm_unreachable("unexpected DeclAnalysis item");
      }
    }
  }

private:
  // Verify a single Decl. Return the pending structure if that was the result,
  // else null. This method must not recurse.
  PendingFunctionAnalysis *verifyDecl(const Decl *D) {
    CallableInfo CInfo(Sem, *D);
    bool isExternC = false;

    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      assert(FD->getBuiltinID() == 0);
      isExternC = FD->getCanonicalDecl()->isExternCContext();
    }

    // For C++, with non-extern "C" linkage only - if any of the Decl's declared
    // effects forbid throwing (e.g. nonblocking) then the function should also
    // be declared noexcept.
    if (Sem.getLangOpts().CPlusPlus && !isExternC) {
      for (const FunctionEffect &Effect : CInfo.Effects) {
        if (!(Effect.flags() & FunctionEffect::FE_ExcludeThrow))
          continue;

        bool IsNoexcept = false;
        if (auto *FD = D->getAsFunction()) {
          IsNoexcept = isNoexcept(FD);
        } else if (auto *BD = dyn_cast<BlockDecl>(D)) {
          if (auto *TSI = BD->getSignatureAsWritten()) {
            auto *FPT = TSI->getType()->getAs<FunctionProtoType>();
            IsNoexcept = FPT->isNothrow() || BD->hasAttr<NoThrowAttr>();
          }
        }
        if (!IsNoexcept)
          Sem.Diag(D->getBeginLoc(),
                   diag::warn_perf_constraint_implies_noexcept)
              << Effect.name();
        break;
      }
    }

    // Build a PendingFunctionAnalysis on the stack. If it turns out to be
    // complete, we'll have avoided a heap allocation; if it's incomplete, it's
    // a fairly trivial move to a heap-allocated object.
    PendingFunctionAnalysis FAnalysis(Sem, CInfo, AllInferrableEffectsToVerify);

    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "\nVerifying " << CInfo.name(Sem) << " ";
      FAnalysis.dump(Sem, llvm::outs());
    }

    FunctionBodyASTVisitor Visitor(*this, FAnalysis, CInfo);

    Visitor.run();
    if (FAnalysis.isComplete()) {
      completeAnalysis(CInfo, FAnalysis);
      return nullptr;
    }
    // Move the pending analysis to the heap and save it in the map.
    PendingFunctionAnalysis *PendingPtr =
        new PendingFunctionAnalysis(std::move(FAnalysis));
    DeclAnalysis[D] = PendingPtr;
    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "inserted pending " << PendingPtr << "\n";
      DeclAnalysis.dump(Sem, llvm::outs());
    }
    return PendingPtr;
  }

  // Consume PendingFunctionAnalysis, create with it a CompleteFunctionAnalysis,
  // inserted in the container.
  void completeAnalysis(const CallableInfo &CInfo,
                        PendingFunctionAnalysis &Pending) {
    if (SmallVector<Diagnostic, 0> &Diags =
            Pending.getDiagnosticsForExplicitFX();
        !Diags.empty())
      emitDiagnostics(Diags, CInfo, Sem);

    CompleteFunctionAnalysis *CompletePtr = new CompleteFunctionAnalysis(
        Sem.getASTContext(), Pending, CInfo.Effects,
        AllInferrableEffectsToVerify);
    DeclAnalysis[CInfo.CDecl] = CompletePtr;
    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "inserted complete " << CompletePtr << "\n";
      DeclAnalysis.dump(Sem, llvm::outs());
    }
  }

  // Called after all direct calls requiring inference have been found -- or
  // not. Repeats calls to FunctionBodyASTVisitor::followCall() but without
  // the possibility of inference. Deletes Pending.
  void finishPendingAnalysis(const Decl *D, PendingFunctionAnalysis *Pending) {
    CallableInfo Caller(Sem, *D);
    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "finishPendingAnalysis for " << Caller.name(Sem) << " : ";
      Pending->dump(Sem, llvm::outs());
      llvm::outs() << "\n";
    }
    for (const PendingFunctionAnalysis::DirectCall &Call :
         Pending->unverifiedCalls()) {
      if (Call.Recursed)
        continue;

      CallableInfo Callee(Sem, *Call.Callee);
      followCall(Caller, *Pending, Callee, Call.CallLoc,
                 /*AssertNoFurtherInference=*/true);
    }
    completeAnalysis(Caller, *Pending);
    delete Pending;
  }

  // Here we have a call to a Decl, either explicitly via a CallExpr or some
  // other AST construct. PFA pertains to the caller.
  void followCall(const CallableInfo &Caller, PendingFunctionAnalysis &PFA,
                  const CallableInfo &Callee, SourceLocation CallLoc,
                  bool AssertNoFurtherInference) {
    const bool DirectCall = Callee.isDirectCall();

    // Initially, the declared effects; inferred effects will be added.
    EffectSet CalleeEffects = Callee.Effects;

    bool IsInferencePossible = DirectCall;

    if (DirectCall) {
      if (CompleteFunctionAnalysis *CFA =
              DeclAnalysis.completedAnalysisForDecl(Callee.CDecl)) {
        // Combine declared effects with those which may have been inferred.
        CalleeEffects.insert(CFA->VerifiedEffects);
        IsInferencePossible = false; // we've already traversed it
      }
    }

    if (AssertNoFurtherInference) {
      assert(!IsInferencePossible);
    }

    if (!Callee.isVerifiable())
      IsInferencePossible = false;

    if constexpr (DebugLogLevel > 0) {
      llvm::outs() << "followCall from " << Caller.name(Sem) << " to "
                   << Callee.name(Sem)
                   << "; verifiable: " << Callee.isVerifiable() << "; callee ";
      CalleeEffects.dump(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "  callee " << Callee.CDecl << " canonical "
                   << CanonicalFunctionDecl(Callee.CDecl) << " redecls";
      for (Decl *D : Callee.CDecl->redecls())
        llvm::outs() << " " << D;

      llvm::outs() << "\n";
    }

    auto check1Effect = [&](const FunctionEffect &Effect, bool Inferring) {
      FunctionEffect::Flags Flags = Effect.flags();
      bool Diagnose =
          Effect.shouldDiagnoseFunctionCall(DirectCall, CalleeEffects);
      if (Diagnose) {
        // If inference is not allowed, or the target is indirect (virtual
        // method/function ptr?), generate a diagnostic now.
        if (!IsInferencePossible ||
            !(Flags & FunctionEffect::FE_InferrableOnCallees)) {
          if (Callee.FuncType == SpecialFuncType::None)
            PFA.checkAddDiagnostic(
                Inferring, {Effect, DiagnosticID::CallsDeclWithoutEffect,
                            CallLoc, Callee.CDecl});
          else
            PFA.checkAddDiagnostic(
                Inferring, {Effect, DiagnosticID::AllocatesMemory, CallLoc});
        } else {
          // Inference is allowed and necessary; defer it.
          PFA.addUnverifiedDirectCall(Callee.CDecl, CallLoc);
        }
      }
    };

    for (const FunctionEffect &Effect : PFA.DeclaredVerifiableEffects)
      check1Effect(Effect, false);

    for (const FunctionEffect &Effect : PFA.FXToInfer)
      check1Effect(Effect, true);
  }

  // Should only be called when determined to be complete.
  void emitDiagnostics(SmallVector<Diagnostic, 0> &Diags,
                       const CallableInfo &CInfo, Sema &S) {
    if (Diags.empty())
      return;
    const SourceManager &SM = S.getSourceManager();
    std::sort(Diags.begin(), Diags.end(),
              [&SM](const Diagnostic &LHS, const Diagnostic &RHS) {
                return SM.isBeforeInTranslationUnit(LHS.Loc, RHS.Loc);
              });

    auto checkAddTemplateNote = [&](const Decl *D) {
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        while (FD != nullptr && FD->isTemplateInstantiation()) {
          S.Diag(FD->getPointOfInstantiation(),
                 diag::note_func_effect_from_template);
          FD = FD->getTemplateInstantiationPattern();
        }
      }
    };

    // Top-level diagnostics are warnings.
    for (const Diagnostic &Diag : Diags) {
      StringRef effectName = Diag.Effect.name();
      switch (Diag.ID) {
      case DiagnosticID::None:
      case DiagnosticID::DeclDisallowsInference: // shouldn't happen
                                                 // here
        llvm_unreachable("Unexpected diagnostic kind");
        break;
      case DiagnosticID::AllocatesMemory:
        S.Diag(Diag.Loc, diag::warn_func_effect_allocates) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::Throws:
      case DiagnosticID::Catches:
        S.Diag(Diag.Loc, diag::warn_func_effect_throws_or_catches)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::HasStaticLocal:
        S.Diag(Diag.Loc, diag::warn_func_effect_has_static_local) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::AccessesThreadLocal:
        S.Diag(Diag.Loc, diag::warn_func_effect_uses_thread_local)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::CallsObjC:
        S.Diag(Diag.Loc, diag::warn_func_effect_calls_objc) << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;
      case DiagnosticID::CallsExprWithoutEffect:
        S.Diag(Diag.Loc, diag::warn_func_effect_calls_expr_without_effect)
            << effectName;
        checkAddTemplateNote(CInfo.CDecl);
        break;

      case DiagnosticID::CallsDeclWithoutEffect: {
        CallableInfo CalleeInfo(S, *Diag.Callee);
        std::string CalleeName = CalleeInfo.name(S);

        S.Diag(Diag.Loc, diag::warn_func_effect_calls_func_without_effect)
            << effectName << CalleeName;
        checkAddTemplateNote(CInfo.CDecl);

        // Emit notes explaining the transitive chain of inferences: Why isn't
        // the callee safe?
        for (const Decl *Callee = Diag.Callee; Callee != nullptr;) {
          std::optional<CallableInfo> MaybeNextCallee;
          CompleteFunctionAnalysis *Completed =
              DeclAnalysis.completedAnalysisForDecl(CalleeInfo.CDecl);
          if (Completed == nullptr) {
            // No result - could be
            // - non-inline
            // - indirect (virtual or through function pointer)
            // - effect has been explicitly disclaimed (e.g. "blocking")
            if (CalleeInfo.CType == CallType::Virtual)
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_virtual)
                  << effectName;
            else if (CalleeInfo.CType == CallType::Unknown)
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_func_ptr)
                  << effectName;
            else if (CalleeInfo.Effects.contains(Diag.Effect.oppositeKind()))
              S.Diag(Callee->getLocation(),
                     diag::note_func_effect_call_disallows_inference)
                  << effectName;
            else
              S.Diag(Callee->getLocation(), diag::note_func_effect_call_extern)
                  << effectName;

            break;
          }
          const Diagnostic *PtrDiag2 =
              Completed->firstDiagnosticForEffect(Diag.Effect);
          if (PtrDiag2 == nullptr)
            break;

          const Diagnostic &Diag2 = *PtrDiag2;
          switch (Diag2.ID) {
          case DiagnosticID::None:
            llvm_unreachable("Unexpected diagnostic kind");
            break;
          case DiagnosticID::DeclDisallowsInference:
            S.Diag(Diag2.Loc, diag::note_func_effect_call_disallows_inference)
                << effectName;
            break;
          case DiagnosticID::CallsExprWithoutEffect:
            S.Diag(Diag2.Loc, diag::note_func_effect_call_func_ptr)
                << effectName;
            break;
          case DiagnosticID::AllocatesMemory:
            S.Diag(Diag2.Loc, diag::note_func_effect_allocates) << effectName;
            break;
          case DiagnosticID::Throws:
          case DiagnosticID::Catches:
            S.Diag(Diag2.Loc, diag::note_func_effect_throws_or_catches)
                << effectName;
            break;
          case DiagnosticID::HasStaticLocal:
            S.Diag(Diag2.Loc, diag::note_func_effect_has_static_local)
                << effectName;
            break;
          case DiagnosticID::AccessesThreadLocal:
            S.Diag(Diag2.Loc, diag::note_func_effect_uses_thread_local)
                << effectName;
            break;
          case DiagnosticID::CallsObjC:
            S.Diag(Diag2.Loc, diag::note_func_effect_calls_objc) << effectName;
            break;
          case DiagnosticID::CallsDeclWithoutEffect:
            MaybeNextCallee.emplace(S, *Diag2.Callee);
            S.Diag(Diag2.Loc, diag::note_func_effect_calls_func_without_effect)
                << effectName << MaybeNextCallee->name(S);
            break;
          }
          checkAddTemplateNote(Callee);
          Callee = Diag2.Callee;
          if (MaybeNextCallee) {
            CalleeInfo = *MaybeNextCallee;
            CalleeName = CalleeInfo.name(S);
          }
        }
      } break;
      }
    }
  }

  // ----------
  // This AST visitor is used to traverse the body of a function during effect
  // verification. This happens in 2 situations:
  //  [1] The function has declared effects which need to be validated.
  //  [2] The function has not explicitly declared an effect in question, and is
  //      being checked for implicit conformance.
  //
  // Diagnostics are always routed to a PendingFunctionAnalysis, which holds
  // all diagnostic output.
  //
  // Q: Currently we create a new RecursiveASTVisitor for every function
  // analysis. Is it so lightweight that this is OK? It would appear so.
  struct FunctionBodyASTVisitor
      : public RecursiveASTVisitor<FunctionBodyASTVisitor> {
    // The meanings of the boolean values returned by the Visit methods can be
    // difficult to remember.
    constexpr static bool Stop = false;
    constexpr static bool Proceed = true;

    Analyzer &Outer;
    PendingFunctionAnalysis &CurrentFunction;
    CallableInfo &CurrentCaller;

    FunctionBodyASTVisitor(Analyzer &outer,
                           PendingFunctionAnalysis &CurrentFunction,
                           CallableInfo &CurrentCaller)
        : Outer(outer), CurrentFunction(CurrentFunction),
          CurrentCaller(CurrentCaller) {}

    // -- Entry point --
    void run() {
      // The target function itself may have some implicit code paths beyond the
      // body: member and base constructors and destructors. Visit these first.
      if (const auto *FD = dyn_cast<const FunctionDecl>(CurrentCaller.CDecl)) {
        if (auto *Ctor = dyn_cast<CXXConstructorDecl>(FD)) {
          for (const CXXCtorInitializer *Initer : Ctor->inits())
            if (Expr *Init = Initer->getInit())
              VisitStmt(Init);
        } else if (auto *Dtor = dyn_cast<CXXDestructorDecl>(FD))
          followDestructor(dyn_cast<CXXRecordDecl>(Dtor->getParent()), Dtor);
      }
      // else could be BlockDecl

      // Do an AST traversal of the function/block body
      TraverseDecl(const_cast<Decl *>(CurrentCaller.CDecl));
    }

    // -- Methods implementing common logic --

    // Handle a language construct forbidden by some effects. Only effects whose
    // flags include the specified flag receive a diagnostic. \p Flag describes
    // the construct.
    void diagnoseLanguageConstruct(FunctionEffect::FlagBit Flag, DiagnosticID D,
                                   SourceLocation Loc,
                                   const Decl *Callee = nullptr) {
      // If there are any declared verifiable effects which forbid the construct
      // represented by the flag, store just one diagnostic.
      for (const FunctionEffect &Effect :
           CurrentFunction.DeclaredVerifiableEffects) {
        if (Effect.flags() & Flag) {
          addDiagnostic(/*inferring=*/false, Effect, D, Loc, Callee);
          break;
        }
      }
      // For each inferred effect which forbids the construct, store a
      // diagnostic, if we don't already have a diagnostic for that effect.
      for (const FunctionEffect &Effect : CurrentFunction.FXToInfer)
        if (Effect.flags() & Flag)
          addDiagnostic(/*inferring=*/true, Effect, D, Loc, Callee);
    }

    void addDiagnostic(bool Inferring, const FunctionEffect &Effect,
                       DiagnosticID D, SourceLocation Loc,
                       const Decl *Callee = nullptr) {
      CurrentFunction.checkAddDiagnostic(Inferring,
                                         Diagnostic(Effect, D, Loc, Callee));
    }

    // Here we have a call to a Decl, either explicitly via a CallExpr or some
    // other AST construct. CallableInfo pertains to the callee.
    void followCall(const CallableInfo &CI, SourceLocation CallLoc) {
      // Currently, built-in functions are always considered safe.
      // FIXME: Some are not.
      if (const auto *FD = dyn_cast<FunctionDecl>(CI.CDecl);
          FD && FD->getBuiltinID() != 0)
        return;

      Outer.followCall(CurrentCaller, CurrentFunction, CI, CallLoc,
                       /*AssertNoFurtherInference=*/false);
    }

    void checkIndirectCall(CallExpr *Call, Expr *CalleeExpr) {
      const QualType CalleeType = CalleeExpr->getType();
      auto *FPT =
          CalleeType->getAs<FunctionProtoType>(); // null if FunctionType
      EffectSet CalleeFX;
      if (FPT)
        CalleeFX.insert(FPT->getFunctionEffects());

      auto check1Effect = [&](const FunctionEffect &Effect, bool Inferring) {
        if (FPT == nullptr || Effect.shouldDiagnoseFunctionCall(
                                  /*direct=*/false, CalleeFX))
          addDiagnostic(Inferring, Effect, DiagnosticID::CallsExprWithoutEffect,
                        Call->getBeginLoc());
      };

      for (const FunctionEffect &Effect :
           CurrentFunction.DeclaredVerifiableEffects)
        check1Effect(Effect, false);

      for (const FunctionEffect &Effect : CurrentFunction.FXToInfer)
        check1Effect(Effect, true);
    }

    // This destructor's body should be followed by the caller, but here we
    // follow the field and base destructors.
    void followDestructor(const CXXRecordDecl *Rec,
                          const CXXDestructorDecl *Dtor) {
      for (const FieldDecl *Field : Rec->fields())
        followTypeDtor(Field->getType());

      if (const auto *Class = dyn_cast<CXXRecordDecl>(Rec)) {
        for (const CXXBaseSpecifier &Base : Class->bases())
          followTypeDtor(Base.getType());

        for (const CXXBaseSpecifier &Base : Class->vbases())
          followTypeDtor(Base.getType());
      }
    }

    void followTypeDtor(QualType QT) {
      const Type *Ty = QT.getTypePtr();
      while (Ty->isArrayType()) {
        const ArrayType *Arr = Ty->getAsArrayTypeUnsafe();
        QT = Arr->getElementType();
        Ty = QT.getTypePtr();
      }

      if (Ty->isRecordType()) {
        if (const CXXRecordDecl *Class = Ty->getAsCXXRecordDecl()) {
          if (CXXDestructorDecl *Dtor = Class->getDestructor()) {
            CallableInfo CI(Outer.Sem, *Dtor);
            followCall(CI, Dtor->getLocation());
          }
        }
      }
    }

    // -- Methods for use of RecursiveASTVisitor --

    bool shouldVisitImplicitCode() const { return true; }

    bool shouldWalkTypesOfTypeLocs() const { return false; }

    bool VisitCXXThrowExpr(CXXThrowExpr *Throw) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThrow,
                                DiagnosticID::Throws, Throw->getThrowLoc());
      return Proceed;
    }

    bool VisitCXXCatchStmt(CXXCatchStmt *Catch) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeCatch,
                                DiagnosticID::Catches, Catch->getCatchLoc());
      return Proceed;
    }

    bool VisitObjCMessageExpr(ObjCMessageExpr *Msg) {
      diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeObjCMessageSend,
                                DiagnosticID::CallsObjC, Msg->getBeginLoc());
      return Proceed;
    }

    bool VisitCallExpr(CallExpr *Call) {
      if constexpr (DebugLogLevel > 2) {
        llvm::errs() << "VisitCallExpr : "
                     << Call->getBeginLoc().printToString(Outer.Sem.SourceMgr)
                     << "\n";
      }

      Expr *CalleeExpr = Call->getCallee();
      if (const Decl *Callee = CalleeExpr->getReferencedDeclOfCallee()) {
        CallableInfo CI(Outer.Sem, *Callee);
        followCall(CI, Call->getBeginLoc());
        return Proceed;
      }

      if (isa<CXXPseudoDestructorExpr>(CalleeExpr))
        // just destroying a scalar, fine.
        return Proceed;

      // No Decl, just an Expr. Just check based on its type.
      checkIndirectCall(Call, CalleeExpr);

      return Proceed;
    }

    bool VisitVarDecl(VarDecl *Var) {
      if constexpr (DebugLogLevel > 2) {
        llvm::errs() << "VisitVarDecl : "
                     << Var->getBeginLoc().printToString(Outer.Sem.SourceMgr)
                     << "\n";
      }

      if (Var->isStaticLocal())
        diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeStaticLocalVars,
                                  DiagnosticID::HasStaticLocal,
                                  Var->getLocation());

      const QualType::DestructionKind DK =
          Var->needsDestruction(Outer.Sem.getASTContext());
      if (DK == QualType::DK_cxx_destructor) {
        QualType QT = Var->getType();
        if (const auto *ClsType = QT.getTypePtr()->getAs<RecordType>()) {
          if (const auto *CxxRec =
                  dyn_cast<CXXRecordDecl>(ClsType->getDecl())) {
            if (const CXXDestructorDecl *Dtor = CxxRec->getDestructor()) {
              CallableInfo CI(Outer.Sem, *Dtor);
              followCall(CI, Var->getLocation());
            }
          }
        }
      }
      return Proceed;
    }

    bool VisitCXXNewExpr(CXXNewExpr *New) {
      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to operator new.
      if (FunctionDecl *FD = New->getOperatorNew()) {
        CallableInfo CI(Outer.Sem, *FD, SpecialFuncType::OperatorNew);
        followCall(CI, New->getBeginLoc());
      }

      // It's a bit excessive to check operator delete here, since it's
      // just a fallback for operator new followed by a failed constructor.
      // We could check it via New->getOperatorDelete().

      // It DOES however visit the called constructor
      return Proceed;
    }

    bool VisitCXXDeleteExpr(CXXDeleteExpr *Delete) {
      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to operator delete.
      if (FunctionDecl *FD = Delete->getOperatorDelete()) {
        CallableInfo CI(Outer.Sem, *FD, SpecialFuncType::OperatorDelete);
        followCall(CI, Delete->getBeginLoc());
      }

      // It DOES however visit the called destructor

      return Proceed;
    }

    bool VisitCXXConstructExpr(CXXConstructExpr *Construct) {
      if constexpr (DebugLogLevel > 2) {
        llvm::errs() << "VisitCXXConstructExpr : "
                     << Construct->getBeginLoc().printToString(
                            Outer.Sem.SourceMgr)
                     << "\n";
      }

      // BUG? It seems incorrect that RecursiveASTVisitor does not
      // visit the call to the constructor.
      const CXXConstructorDecl *Ctor = Construct->getConstructor();
      CallableInfo CI(Outer.Sem, *Ctor);
      followCall(CI, Construct->getLocation());

      return Proceed;
    }

    bool VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DEI) {
      if (Expr *E = DEI->getExpr())
        TraverseStmt(E);

      return Proceed;
    }

    bool TraverseLambdaExpr(LambdaExpr *Lambda) {
      // We override this so as the be able to skip traversal of the lambda's
      // body. We have to explicitly traverse the captures.
      for (unsigned I = 0, N = Lambda->capture_size(); I < N; ++I)
        if (TraverseLambdaCapture(Lambda, Lambda->capture_begin() + I,
                                  Lambda->capture_init_begin()[I]) == Stop)
          return Stop;

      return Proceed;
    }

    bool TraverseBlockExpr(BlockExpr * /*unused*/) {
      // TODO: are the capture expressions (ctor call?) safe?
      return Proceed;
    }

    bool VisitDeclRefExpr(const DeclRefExpr *E) {
      const ValueDecl *Val = E->getDecl();
      if (isa<VarDecl>(Val)) {
        const VarDecl *Var = cast<VarDecl>(Val);
        VarDecl::TLSKind TLSK = Var->getTLSKind();
        if (TLSK != VarDecl::TLS_None) {
          // At least on macOS, thread-local variables are initialized on
          // first access, including a heap allocation.
          diagnoseLanguageConstruct(FunctionEffect::FE_ExcludeThreadLocalVars,
                                    DiagnosticID::AccessesThreadLocal,
                                    E->getLocation());
        }
      }
      return Proceed;
    }

    // Unevaluated contexts: need to skip
    // see https://reviews.llvm.org/rG777eb4bcfc3265359edb7c979d3e5ac699ad4641

    bool TraverseGenericSelectionExpr(GenericSelectionExpr *Node) {
      return TraverseStmt(Node->getResultExpr());
    }
    bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
      return Proceed;
    }

    bool TraverseTypeOfExprTypeLoc(TypeOfExprTypeLoc Node) { return Proceed; }

    bool TraverseDecltypeTypeLoc(DecltypeTypeLoc Node) { return Proceed; }

    bool TraverseCXXNoexceptExpr(CXXNoexceptExpr *Node) { return Proceed; }

    bool TraverseCXXTypeidExpr(CXXTypeidExpr *Node) { return Proceed; }
  };
};

Analyzer::AnalysisMap::~AnalysisMap() {
  for (const auto &Item : *this) {
    FuncAnalysisPtr AP = Item.second;
    if (isa<PendingFunctionAnalysis *>(AP))
      delete AP.get<PendingFunctionAnalysis *>();
    else
      delete AP.get<CompleteFunctionAnalysis *>();
  }
}

} // namespace FXAnalysis

// =============================================================================

//===----------------------------------------------------------------------===//
// AnalysisBasedWarnings - Worker object used by Sema to execute analysis-based
//  warnings on a function, method, or block.
//===----------------------------------------------------------------------===//

sema::AnalysisBasedWarnings::Policy::Policy() {
  enableCheckFallThrough = 1;
  enableCheckUnreachable = 0;
  enableThreadSafetyAnalysis = 0;
  enableConsumedAnalysis = 0;
}

/// InterProceduralData aims to be a storage of whatever data should be passed
/// between analyses of different functions.
///
/// At the moment, its primary goal is to make the information gathered during
/// the analysis of the blocks available during the analysis of the enclosing
/// function.  This is important due to the fact that blocks are analyzed before
/// the enclosed function is even parsed fully, so it is not viable to access
/// anything in the outer scope while analyzing the block.  On the other hand,
/// re-building CFG for blocks and re-analyzing them when we do have all the
/// information (i.e. during the analysis of the enclosing function) seems to be
/// ill-designed.
class sema::AnalysisBasedWarnings::InterProceduralData {
public:
  // It is important to analyze blocks within functions because it's a very
  // common pattern to capture completion handler parameters by blocks.
  CalledOnceInterProceduralData CalledOnceData;
};

static unsigned isEnabled(DiagnosticsEngine &D, unsigned diag) {
  return (unsigned)!D.isIgnored(diag, SourceLocation());
}

sema::AnalysisBasedWarnings::AnalysisBasedWarnings(Sema &s)
    : S(s), IPData(std::make_unique<InterProceduralData>()),
      NumFunctionsAnalyzed(0), NumFunctionsWithBadCFGs(0), NumCFGBlocks(0),
      MaxCFGBlocksPerFunction(0), NumUninitAnalysisFunctions(0),
      NumUninitAnalysisVariables(0), MaxUninitAnalysisVariablesPerFunction(0),
      NumUninitAnalysisBlockVisits(0),
      MaxUninitAnalysisBlockVisitsPerFunction(0) {

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
sema::AnalysisBasedWarnings::~AnalysisBasedWarnings() = default;

static void flushDiagnostics(Sema &S, const sema::FunctionScopeInfo *fscope) {
  for (const auto &D : fscope->PossiblyUnreachableDiags)
    S.Diag(D.Loc, D.PD);
}

// An AST Visitor that calls a callback function on each callable DEFINITION
// that is NOT in a dependent context:
class CallableVisitor : public RecursiveASTVisitor<CallableVisitor> {
private:
  llvm::function_ref<void(const Decl *)> Callback;

public:
  CallableVisitor(llvm::function_ref<void(const Decl *)> Callback)
      : Callback(Callback) {}

  bool VisitFunctionDecl(FunctionDecl *Node) {
    if (cast<DeclContext>(Node)->isDependentContext())
      return true; // Not to analyze dependent decl
    // `FunctionDecl->hasBody()` returns true if the function has a body
    // somewhere defined.  But we want to know if this `Node` has a body
    // child.  So we use `doesThisDeclarationHaveABody`:
    if (Node->doesThisDeclarationHaveABody())
      Callback(Node);
    return true;
  }

  bool VisitBlockDecl(BlockDecl *Node) {
    if (cast<DeclContext>(Node)->isDependentContext())
      return true; // Not to analyze dependent decl
    Callback(Node);
    return true;
  }

  bool VisitObjCMethodDecl(ObjCMethodDecl *Node) {
    if (cast<DeclContext>(Node)->isDependentContext())
      return true; // Not to analyze dependent decl
    if (Node->hasBody())
      Callback(Node);
    return true;
  }

  bool VisitLambdaExpr(LambdaExpr *Node) {
    return VisitFunctionDecl(Node->getCallOperator());
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return false; }
};

void clang::sema::AnalysisBasedWarnings::IssueWarnings(
     TranslationUnitDecl *TU) {
  if (!TU)
    return; // This is unexpected, give up quietly.

  DiagnosticsEngine &Diags = S.getDiagnostics();

  if (S.hasUncompilableErrorOccurred() || Diags.getIgnoreAllWarnings())
    // exit if having uncompilable errors or ignoring all warnings:
    return;

  DiagnosticOptions &DiagOpts = Diags.getDiagnosticOptions();

  // UnsafeBufferUsage analysis settings.
  bool UnsafeBufferUsageCanEmitSuggestions = S.getLangOpts().CPlusPlus20;
  bool UnsafeBufferUsageShouldEmitSuggestions =  // Should != Can.
      UnsafeBufferUsageCanEmitSuggestions &&
      DiagOpts.ShowSafeBufferUsageSuggestions;
  bool UnsafeBufferUsageShouldSuggestSuggestions =
      UnsafeBufferUsageCanEmitSuggestions &&
      !DiagOpts.ShowSafeBufferUsageSuggestions;
  UnsafeBufferUsageReporter R(S, UnsafeBufferUsageShouldSuggestSuggestions);

  // The Callback function that performs analyses:
  auto CallAnalyzers = [&](const Decl *Node) -> void {
    // Perform unsafe buffer usage analysis:
    if (!Diags.isIgnored(diag::warn_unsafe_buffer_operation,
                         Node->getBeginLoc()) ||
        !Diags.isIgnored(diag::warn_unsafe_buffer_variable,
                         Node->getBeginLoc()) ||
        !Diags.isIgnored(diag::warn_unsafe_buffer_usage_in_container,
                         Node->getBeginLoc())) {
      clang::checkUnsafeBufferUsage(Node, R,
                                    UnsafeBufferUsageShouldEmitSuggestions);
    }

    // More analysis ...
  };
  // Emit per-function analysis-based warnings that require the whole-TU
  // reasoning. Check if any of them is enabled at all before scanning the AST:
  if (!Diags.isIgnored(diag::warn_unsafe_buffer_operation, SourceLocation()) ||
      !Diags.isIgnored(diag::warn_unsafe_buffer_variable, SourceLocation()) ||
      !Diags.isIgnored(diag::warn_unsafe_buffer_usage_in_container,
                       SourceLocation())) {
    CallableVisitor(CallAnalyzers).TraverseTranslationUnitDecl(TU);
  }

  if (S.Context.hasAnyFunctionEffects())
    FXAnalysis::Analyzer{S}.run(*TU);
}

void clang::sema::AnalysisBasedWarnings::IssueWarnings(
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

  // Construct the analysis context with the specified CFG build options.
  AnalysisDeclContext AC(/* AnalysisDeclContextManager */ nullptr, D);

  // Don't generate EH edges for CallExprs as we'd like to avoid the n^2
  // explosion for destructors that can result and the compile time hit.
  AC.getCFGBuildOptions().PruneTriviallyFalseEdges = true;
  AC.getCFGBuildOptions().AddEHEdges = false;
  AC.getCFGBuildOptions().AddInitializers = true;
  AC.getCFGBuildOptions().AddImplicitDtors = true;
  AC.getCFGBuildOptions().AddTemporaryDtors = true;
  AC.getCFGBuildOptions().AddCXXNewAllocator = false;
  AC.getCFGBuildOptions().AddCXXDefaultInitExprInCtors = true;

  // Force that certain expressions appear as CFGElements in the CFG.  This
  // is used to speed up various analyses.
  // FIXME: This isn't the right factoring.  This is here for initial
  // prototyping, but we need a way for analyses to say what expressions they
  // expect to always be CFGElements and then fill in the BuildOptions
  // appropriately.  This is essentially a layering violation.
  if (P.enableCheckUnreachable || P.enableThreadSafetyAnalysis ||
      P.enableConsumedAnalysis) {
    // Unreachable code analysis and thread safety require a linearized CFG.
    AC.getCFGBuildOptions().setAllAlwaysAdd();
  }
  else {
    AC.getCFGBuildOptions()
      .setAlwaysAdd(Stmt::BinaryOperatorClass)
      .setAlwaysAdd(Stmt::CompoundAssignOperatorClass)
      .setAlwaysAdd(Stmt::BlockExprClass)
      .setAlwaysAdd(Stmt::CStyleCastExprClass)
      .setAlwaysAdd(Stmt::DeclRefExprClass)
      .setAlwaysAdd(Stmt::ImplicitCastExprClass)
      .setAlwaysAdd(Stmt::UnaryOperatorClass);
  }

  // Install the logical handler.
  std::optional<LogicalErrorHandler> LEH;
  if (LogicalErrorHandler::hasActiveDiagnostics(Diags, D->getBeginLoc())) {
    LEH.emplace(S);
    AC.getCFGBuildOptions().Observer = &*LEH;
  }

  // Emit delayed diagnostics.
  if (!fscope->PossiblyUnreachableDiags.empty()) {
    bool analyzed = false;

    // Register the expressions with the CFGBuilder.
    for (const auto &D : fscope->PossiblyUnreachableDiags) {
      for (const Stmt *S : D.Stmts)
        AC.registerForcedBlockExpression(S);
    }

    if (AC.getCFG()) {
      analyzed = true;
      for (const auto &D : fscope->PossiblyUnreachableDiags) {
        bool AllReachable = true;
        for (const Stmt *S : D.Stmts) {
          const CFGBlock *block = AC.getBlockForRegisteredExpression(S);
          CFGReverseBlockReachabilityAnalysis *cra =
              AC.getCFGReachablityAnalysis();
          // FIXME: We should be able to assert that block is non-null, but
          // the CFG analysis can skip potentially-evaluated expressions in
          // edge cases; see test/Sema/vla-2.c.
          if (block && cra) {
            // Can this block be reached from the entrance?
            if (!cra->isReachable(&AC.getCFG()->getEntry(), block)) {
              AllReachable = false;
              break;
            }
          }
          // If we cannot map to a basic block, assume the statement is
          // reachable.
        }

        if (AllReachable)
          S.Diag(D.Loc, D.PD);
      }
    }

    if (!analyzed)
      flushDiagnostics(S, fscope);
  }

  // Warning: check missing 'return'
  if (P.enableCheckFallThrough) {
    const CheckFallThroughDiagnostics &CD =
        (isa<BlockDecl>(D)
             ? CheckFallThroughDiagnostics::MakeForBlock()
             : (isa<CXXMethodDecl>(D) &&
                cast<CXXMethodDecl>(D)->getOverloadedOperator() == OO_Call &&
                cast<CXXMethodDecl>(D)->getParent()->isLambda())
                   ? CheckFallThroughDiagnostics::MakeForLambda()
                   : (fscope->isCoroutine()
                          ? CheckFallThroughDiagnostics::MakeForCoroutine(D)
                          : CheckFallThroughDiagnostics::MakeForFunction(D)));
    CheckFallThroughForBody(S, D, Body, BlockType, CD, AC, fscope);
  }

  // Warning: check for unreachable code
  if (P.enableCheckUnreachable) {
    // Only check for unreachable code on non-template instantiations.
    // Different template instantiations can effectively change the control-flow
    // and it is very difficult to prove that a snippet of code in a template
    // is unreachable for all instantiations.
    bool isTemplateInstantiation = false;
    if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(D))
      isTemplateInstantiation = Function->isTemplateInstantiation();
    if (!isTemplateInstantiation)
      CheckUnreachable(S, AC);
  }

  // Check for thread safety violations
  if (P.enableThreadSafetyAnalysis) {
    SourceLocation FL = AC.getDecl()->getLocation();
    SourceLocation FEL = AC.getDecl()->getEndLoc();
    threadSafety::ThreadSafetyReporter Reporter(S, FL, FEL);
    if (!Diags.isIgnored(diag::warn_thread_safety_beta, D->getBeginLoc()))
      Reporter.setIssueBetaWarnings(true);
    if (!Diags.isIgnored(diag::warn_thread_safety_verbose, D->getBeginLoc()))
      Reporter.setVerbose(true);

    threadSafety::runThreadSafetyAnalysis(AC, Reporter,
                                          &S.ThreadSafetyDeclCache);
    Reporter.emitDiagnostics();
  }

  // Check for violations of consumed properties.
  if (P.enableConsumedAnalysis) {
    consumed::ConsumedWarningsHandler WarningHandler(S);
    consumed::ConsumedAnalyzer Analyzer(WarningHandler);
    Analyzer.run(AC);
  }

  if (!Diags.isIgnored(diag::warn_uninit_var, D->getBeginLoc()) ||
      !Diags.isIgnored(diag::warn_sometimes_uninit_var, D->getBeginLoc()) ||
      !Diags.isIgnored(diag::warn_maybe_uninit_var, D->getBeginLoc()) ||
      !Diags.isIgnored(diag::warn_uninit_const_reference, D->getBeginLoc())) {
    if (CFG *cfg = AC.getCFG()) {
      UninitValsDiagReporter reporter(S);
      UninitVariablesAnalysisStats stats;
      std::memset(&stats, 0, sizeof(UninitVariablesAnalysisStats));
      runUninitializedVariablesAnalysis(*cast<DeclContext>(D), *cfg, AC,
                                        reporter, stats);

      if (S.CollectStats && stats.NumVariablesAnalyzed > 0) {
        ++NumUninitAnalysisFunctions;
        NumUninitAnalysisVariables += stats.NumVariablesAnalyzed;
        NumUninitAnalysisBlockVisits += stats.NumBlockVisits;
        MaxUninitAnalysisVariablesPerFunction =
            std::max(MaxUninitAnalysisVariablesPerFunction,
                     stats.NumVariablesAnalyzed);
        MaxUninitAnalysisBlockVisitsPerFunction =
            std::max(MaxUninitAnalysisBlockVisitsPerFunction,
                     stats.NumBlockVisits);
      }
    }
  }

  // Check for violations of "called once" parameter properties.
  if (S.getLangOpts().ObjC && !S.getLangOpts().CPlusPlus &&
      shouldAnalyzeCalledOnceParameters(Diags, D->getBeginLoc())) {
    if (AC.getCFG()) {
      CalledOnceCheckReporter Reporter(S, IPData->CalledOnceData);
      checkCalledOnceParameters(
          AC, Reporter,
          shouldAnalyzeCalledOnceConventions(Diags, D->getBeginLoc()));
    }
  }

  bool FallThroughDiagFull =
      !Diags.isIgnored(diag::warn_unannotated_fallthrough, D->getBeginLoc());
  bool FallThroughDiagPerFunction = !Diags.isIgnored(
      diag::warn_unannotated_fallthrough_per_function, D->getBeginLoc());
  if (FallThroughDiagFull || FallThroughDiagPerFunction ||
      fscope->HasFallthroughStmt) {
    DiagnoseSwitchLabelsFallthrough(S, AC, !FallThroughDiagFull);
  }

  if (S.getLangOpts().ObjCWeak &&
      !Diags.isIgnored(diag::warn_arc_repeated_use_of_weak, D->getBeginLoc()))
    diagnoseRepeatedUseOfWeak(S, fscope, D, AC.getParentMap());


  // Check for infinite self-recursion in functions
  if (!Diags.isIgnored(diag::warn_infinite_recursive_function,
                       D->getBeginLoc())) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      checkRecursiveFunction(S, FD, Body, AC);
    }
  }

  // Check for throw out of non-throwing function.
  if (!Diags.isIgnored(diag::warn_throw_in_noexcept_func, D->getBeginLoc()))
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (S.getLangOpts().CPlusPlus && !fscope->isCoroutine() && isNoexcept(FD))
        checkThrowInNonThrowingFunc(S, FD, AC);

  // If none of the previous checks caused a CFG build, trigger one here
  // for the logical error handler.
  if (LogicalErrorHandler::hasActiveDiagnostics(Diags, D->getBeginLoc())) {
    AC.getCFG();
  }

  // Collect statistics about the CFG if it was built.
  if (S.CollectStats && AC.isCFGBuilt()) {
    ++NumFunctionsAnalyzed;
    if (CFG *cfg = AC.getCFG()) {
      // If we successfully built a CFG for this context, record some more
      // detail information about it.
      NumCFGBlocks += cfg->getNumBlockIDs();
      MaxCFGBlocksPerFunction = std::max(MaxCFGBlocksPerFunction,
                                         cfg->getNumBlockIDs());
    } else {
      ++NumFunctionsWithBadCFGs;
    }
  }
}

void clang::sema::AnalysisBasedWarnings::PrintStats() const {
  llvm::errs() << "\n*** Analysis Based Warnings Stats:\n";

  unsigned NumCFGsBuilt = NumFunctionsAnalyzed - NumFunctionsWithBadCFGs;
  unsigned AvgCFGBlocksPerFunction =
      !NumCFGsBuilt ? 0 : NumCFGBlocks/NumCFGsBuilt;
  llvm::errs() << NumFunctionsAnalyzed << " functions analyzed ("
               << NumFunctionsWithBadCFGs << " w/o CFGs).\n"
               << "  " << NumCFGBlocks << " CFG blocks built.\n"
               << "  " << AvgCFGBlocksPerFunction
               << " average CFG blocks per function.\n"
               << "  " << MaxCFGBlocksPerFunction
               << " max CFG blocks per function.\n";

  unsigned AvgUninitVariablesPerFunction = !NumUninitAnalysisFunctions ? 0
      : NumUninitAnalysisVariables/NumUninitAnalysisFunctions;
  unsigned AvgUninitBlockVisitsPerFunction = !NumUninitAnalysisFunctions ? 0
      : NumUninitAnalysisBlockVisits/NumUninitAnalysisFunctions;
  llvm::errs() << NumUninitAnalysisFunctions
               << " functions analyzed for uninitialiazed variables\n"
               << "  " << NumUninitAnalysisVariables << " variables analyzed.\n"
               << "  " << AvgUninitVariablesPerFunction
               << " average variables per function.\n"
               << "  " << MaxUninitAnalysisVariablesPerFunction
               << " max variables per function.\n"
               << "  " << NumUninitAnalysisBlockVisits << " block visits.\n"
               << "  " << AvgUninitBlockVisitsPerFunction
               << " average block visits per function.\n"
               << "  " << MaxUninitAnalysisBlockVisitsPerFunction
               << " max block visits per function.\n";
}
