//===--- State.h - State chain for the VM and AST Walker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the base class of the interpreter and evaluator state.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_STATE_H
#define LLVM_CLANG_AST_INTERP_STATE_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OptionalDiagnostic.h"

namespace clang {
class OptionalDiagnostic;

/// Kinds of access we can perform on an object, for diagnostics. Note that
/// we consider a member function call to be a kind of access, even though
/// it is not formally an access of the object, because it has (largely) the
/// same set of semantic restrictions.
enum AccessKinds {
  AK_Read,
  AK_ReadObjectRepresentation,
  AK_Assign,
  AK_Increment,
  AK_Decrement,
  AK_MemberCall,
  AK_DynamicCast,
  AK_TypeId,
  AK_Construct,
  AK_Destroy,
  AK_IsWithinLifetime,
  AK_Dereference
};

/// The order of this enum is important for diagnostics.
enum CheckSubobjectKind {
  CSK_Base,
  CSK_Derived,
  CSK_Field,
  CSK_ArrayToPointer,
  CSK_ArrayIndex,
  CSK_Real,
  CSK_Imag,
  CSK_VectorElement
};

enum class EvaluationMode {
  /// Evaluate as a constant expression. Stop if we find that the expression
  /// is not a constant expression.
  ConstantExpression,

  /// Evaluate as a constant expression. Stop if we find that the expression
  /// is not a constant expression. Some expressions can be retried in the
  /// optimizer if we don't constant fold them here, but in an unevaluated
  /// context we try to fold them immediately since the optimizer never
  /// gets a chance to look at it.
  ConstantExpressionUnevaluated,

  /// Fold the expression to a constant. Stop if we hit a side-effect that
  /// we can't model.
  ConstantFold,

  /// Evaluate in any way we know how. Don't worry about side-effects that
  /// can't be modeled.
  IgnoreSideEffects,
};

namespace interp {
class Frame;
class SourceInfo;

/// Interface for the VM to interact with the AST walker's context.
class State {
public:
  State(ASTContext &ASTCtx, Expr::EvalStatus &EvalStatus)
      : Ctx(ASTCtx), EvalStatus(EvalStatus) {}
  virtual ~State();

  virtual const Frame *getCurrentFrame() = 0;
  virtual const Frame *getBottomFrame() const = 0;
  virtual unsigned getCallStackDepth() = 0;
  virtual bool stepsLeft() const = 0;

  Expr::EvalStatus &getEvalStatus() const { return EvalStatus; }
  ASTContext &getASTContext() const { return Ctx; }
  const LangOptions &getLangOpts() const { return Ctx.getLangOpts(); }

  /// Note that we have had a side-effect, and determine whether we should
  /// keep evaluating.
  bool noteSideEffect() const {
    EvalStatus.HasSideEffects = true;
    return keepEvaluatingAfterSideEffect();
  }

  /// Should we continue evaluation as much as possible after encountering a
  /// construct which can't be reduced to a value?
  bool keepEvaluatingAfterFailure() const;
  /// Should we continue evaluation after encountering a side-effect that we
  /// couldn't model?
  bool keepEvaluatingAfterSideEffect() const;

  /// Note that we hit something that was technically undefined behavior, but
  /// that we can evaluate past it (such as signed overflow or floating-point
  /// division by zero.)
  bool noteUndefinedBehavior() const {
    EvalStatus.HasUndefinedBehavior = true;
    return keepEvaluatingAfterUndefinedBehavior();
  }

  /// Are we checking whether the expression is a potential constant
  /// expression?
  bool checkingPotentialConstantExpression() const {
    return CheckingPotentialConstantExpression;
  }
  /// Are we checking an expression for overflow?
  bool checkingForUndefinedBehavior() const {
    return CheckingForUndefinedBehavior;
  }

  /// Diagnose that the evaluation could not be folded (FF => FoldFailure)
  OptionalDiagnostic
  FFDiag(SourceLocation Loc,
         diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
         unsigned ExtraNotes = 0);

  OptionalDiagnostic
  FFDiag(const Expr *E,
         diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
         unsigned ExtraNotes = 0);

  OptionalDiagnostic
  FFDiag(SourceInfo SI,
         diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
         unsigned ExtraNotes = 0);

  /// Diagnose that the evaluation does not produce a C++11 core constant
  /// expression.
  ///
  /// FIXME: Stop evaluating if we're in EM_ConstantExpression or
  /// EM_PotentialConstantExpression mode and we produce one of these.
  OptionalDiagnostic
  CCEDiag(SourceLocation Loc,
          diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
          unsigned ExtraNotes = 0);

  OptionalDiagnostic
  CCEDiag(const Expr *E,
          diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
          unsigned ExtraNotes = 0);

  OptionalDiagnostic
  CCEDiag(SourceInfo SI,
          diag::kind DiagId = diag::note_invalid_subexpr_in_const_expr,
          unsigned ExtraNotes = 0);

  /// Add a note to a prior diagnostic.
  OptionalDiagnostic Note(SourceLocation Loc, diag::kind DiagId);

  /// Add a stack of notes to a prior diagnostic.
  void addNotes(ArrayRef<PartialDiagnosticAt> Diags);

  /// Directly reports a diagnostic message.
  DiagnosticBuilder report(SourceLocation Loc, diag::kind DiagId);

  /// Whether or not we're in a context where the front end requires a
  /// constant value.
  bool InConstantContext = false;

  /// Whether we're checking that an expression is a potential constant
  /// expression. If so, do not fail on constructs that could become constant
  /// later on (such as a use of an undefined global).
  bool CheckingPotentialConstantExpression = false;

  /// Whether we're checking for an expression that has undefined behavior.
  /// If so, we will produce warnings if we encounter an operation that is
  /// always undefined.
  ///
  /// Note that we still need to evaluate the expression normally when this
  /// is set; this is used when evaluating ICEs in C.
  bool CheckingForUndefinedBehavior = false;

  EvaluationMode EvalMode;
  ASTContext &Ctx;
  Expr::EvalStatus &EvalStatus;

private:
  /// HasActiveDiagnostic - Was the previous diagnostic stored? If so, further
  /// notes attached to it will also be stored, otherwise they will not be.
  bool HasActiveDiagnostic = false;

  /// Have we emitted a diagnostic explaining why we couldn't constant
  /// fold (not just why it's not strictly a constant expression)?
  bool HasFoldFailureDiagnostic = false;

  void addCallStack(unsigned Limit);

  PartialDiagnostic &addDiag(SourceLocation Loc, diag::kind DiagId);

  OptionalDiagnostic diag(SourceLocation Loc, diag::kind DiagId,
                          unsigned ExtraNotes, bool IsCCEDiag);

  /// Should we continue evaluation after encountering undefined behavior?
  bool keepEvaluatingAfterUndefinedBehavior() const;

  // If we have a prior diagnostic, it will be noting that the expression
  // isn't a constant expression. This diagnostic is more important,
  // unless we require this evaluation to produce a constant expression.
  //
  // FIXME: We might want to show both diagnostics to the user in
  // EvaluationMode::ConstantFold mode.
  bool hasPriorDiagnostic();

  void setFoldFailureDiagnostic(bool Flag) { HasFoldFailureDiagnostic = Flag; };
  void setActiveDiagnostic(bool Flag) { HasActiveDiagnostic = Flag; };
  bool hasActiveDiagnostic() const { return HasActiveDiagnostic; }
};

} // namespace interp
} // namespace clang

#endif
