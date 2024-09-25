//===--- InterpState.h - Interpreter state for the constexpr VM -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the interpreter state and entry point.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTERPSTATE_H
#define LLVM_CLANG_AST_INTERP_INTERPSTATE_H

#include "Context.h"
#include "DynamicAllocator.h"
#include "Function.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "State.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OptionalDiagnostic.h"

namespace clang {
namespace interp {
class Context;
class Function;
class InterpStack;
class InterpFrame;
class SourceMapper;

/// Interpreter context.
class InterpState final : public State, public SourceMapper {
public:
  InterpState(State &Parent, Program &P, InterpStack &Stk, Context &Ctx,
              SourceMapper *M = nullptr);

  ~InterpState();

  void cleanup();

  InterpState(const InterpState &) = delete;
  InterpState &operator=(const InterpState &) = delete;

  // Stack frame accessors.
  Frame *getSplitFrame() { return Parent.getCurrentFrame(); }
  Frame *getCurrentFrame() override;
  unsigned getCallStackDepth() override {
    return Current ? (Current->getDepth() + 1) : 1;
  }
  const Frame *getBottomFrame() const override {
    return Parent.getBottomFrame();
  }

  // Access objects from the walker context.
  Expr::EvalStatus &getEvalStatus() const override {
    return Parent.getEvalStatus();
  }
  ASTContext &getASTContext() const override { return Parent.getASTContext(); }

  // Forward status checks and updates to the walker.
  bool checkingForUndefinedBehavior() const override {
    return Parent.checkingForUndefinedBehavior();
  }
  bool keepEvaluatingAfterFailure() const override {
    return Parent.keepEvaluatingAfterFailure();
  }
  bool keepEvaluatingAfterSideEffect() const override {
    return Parent.keepEvaluatingAfterSideEffect();
  }
  bool checkingPotentialConstantExpression() const override {
    return Parent.checkingPotentialConstantExpression();
  }
  bool noteUndefinedBehavior() override {
    return Parent.noteUndefinedBehavior();
  }
  bool inConstantContext() const;
  bool hasActiveDiagnostic() override { return Parent.hasActiveDiagnostic(); }
  void setActiveDiagnostic(bool Flag) override {
    Parent.setActiveDiagnostic(Flag);
  }
  void setFoldFailureDiagnostic(bool Flag) override {
    Parent.setFoldFailureDiagnostic(Flag);
  }
  bool hasPriorDiagnostic() override { return Parent.hasPriorDiagnostic(); }
  bool noteSideEffect() override { return Parent.noteSideEffect(); }

  /// Reports overflow and return true if evaluation should continue.
  bool reportOverflow(const Expr *E, const llvm::APSInt &Value);

  /// Deallocates a pointer.
  void deallocate(Block *B);

  /// Delegates source mapping to the mapper.
  SourceInfo getSource(const Function *F, CodePtr PC) const override {
    if (M)
      return M->getSource(F, PC);

    assert(F && "Function cannot be null");
    return F->getSource(PC);
  }

  Context &getContext() const { return Ctx; }

  void setEvalLocation(SourceLocation SL) { this->EvalLocation = SL; }

  DynamicAllocator &getAllocator() { return Alloc; }

  /// Diagnose any dynamic allocations that haven't been freed yet.
  /// Will return \c false if there were any allocations to diagnose,
  /// \c true otherwise.
  bool maybeDiagnoseDanglingAllocations();

private:
  friend class EvaluationResult;
  friend class InterpStateCCOverride;
  /// AST Walker state.
  State &Parent;
  /// Dead block chain.
  DeadBlock *DeadBlocks = nullptr;
  /// Reference to the offset-source mapping.
  SourceMapper *M;
  /// Allocator used for dynamic allocations performed via the program.
  DynamicAllocator Alloc;
  std::optional<bool> ConstantContextOverride;

public:
  /// Reference to the module containing all bytecode.
  Program &P;
  /// Temporary stack.
  InterpStack &Stk;
  /// Interpreter Context.
  Context &Ctx;
  /// The current frame.
  InterpFrame *Current = nullptr;
  /// Source location of the evaluating expression
  SourceLocation EvalLocation;
  /// Declaration we're initializing/evaluting, if any.
  const VarDecl *EvaluatingDecl = nullptr;

  llvm::SmallVector<
      std::pair<const Expr *, const LifetimeExtendedTemporaryDecl *>>
      SeenGlobalTemporaries;
};

class InterpStateCCOverride final {
public:
  InterpStateCCOverride(InterpState &Ctx, bool Value)
      : Ctx(Ctx), OldCC(Ctx.ConstantContextOverride) {
    // We only override this if the new value is true.
    Enabled = Value;
    if (Enabled)
      Ctx.ConstantContextOverride = Value;
  }
  ~InterpStateCCOverride() {
    if (Enabled)
      Ctx.ConstantContextOverride = OldCC;
  }

private:
  bool Enabled;
  InterpState &Ctx;
  std::optional<bool> OldCC;
};

} // namespace interp
} // namespace clang

#endif
