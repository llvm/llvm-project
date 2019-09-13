//===--- EvalEmitter.h - Instruction emitter for the VM ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the instruction emitters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_EVALEMITTER_H
#define LLVM_CLANG_AST_INTERP_EVALEMITTER_H

#include "ByteCodeGenError.h"
#include "Context.h"
#include "InterpStack.h"
#include "InterpState.h"
#include "PrimType.h"
#include "Program.h"
#include "Source.h"
#include "llvm/Support/Error.h"

namespace clang {
class FunctionDecl;
namespace interp {
class Context;
class Function;
class InterpState;
class Program;
class SourceInfo;
enum Opcode : uint32_t;

/// An emitter which evaluates opcodes as they are emitted.
class EvalEmitter : public SourceMapper {
public:
  using LabelTy = uint32_t;
  using AddrTy = uintptr_t;
  using Local = Scope::Local;

  llvm::Expected<bool> interpretExpr(const Expr *E);
  llvm::Expected<bool> interpretDecl(const VarDecl *VD);

protected:
  EvalEmitter(Context &Ctx, Program &P, State &Parent, InterpStack &Stk,
              APValue &Result);

  virtual ~EvalEmitter() {}

  /// Define a label.
  void emitLabel(LabelTy Label);
  /// Create a label.
  LabelTy getLabel();

  /// Methods implemented by the compiler.
  virtual bool visitExpr(const Expr *E) = 0;
  virtual bool visitDecl(const VarDecl *VD) = 0;

  bool bail(const Stmt *S) { return bail(S->getBeginLoc()); }
  bool bail(const Decl *D) { return bail(D->getBeginLoc()); }
  bool bail(const SourceLocation &Loc);

  /// Emits jumps.
  bool jumpTrue(const LabelTy &Label);
  bool jumpFalse(const LabelTy &Label);
  bool jump(const LabelTy &Label);
  bool fallthrough(const LabelTy &Label);

  /// Callback for registering a local.
  Local createLocal(Descriptor *D);

  /// Returns the source location of the current opcode.
  SourceInfo getSource(Function *F, CodePtr PC) const override {
    return F ? F->getSource(PC) : CurrentSource;
  }

  /// Parameter indices.
  llvm::DenseMap<const ParmVarDecl *, unsigned> Params;
  /// Local descriptors.
  llvm::SmallVector<SmallVector<Local, 8>, 2> Descriptors;

private:
  /// Current compilation context.
  Context &Ctx;
  /// Current program.
  Program &P;
  /// Callee evaluation state.
  InterpState S;
  /// Location to write the result to.
  APValue &Result;

  /// Temporaries which require storage.
  llvm::DenseMap<unsigned, std::unique_ptr<char[]>> Locals;

  // The emitter always tracks the current instruction and sets OpPC to a token
  // value which is mapped to the location of the opcode being evaluated.
  CodePtr OpPC;
  /// Location of a failure.
  llvm::Optional<SourceLocation> BailLocation;
  /// Location of the current instruction.
  SourceInfo CurrentSource;

  /// Next label ID to generate - first label is 1.
  LabelTy NextLabel = 1;
  /// Label being executed - 0 is the entry label.
  LabelTy CurrentLabel = 0;
  /// Active block which should be executed.
  LabelTy ActiveLabel = 0;

  /// Since expressions can only jump forward, predicated execution is
  /// used to deal with if-else statements.
  bool isActive() { return CurrentLabel == ActiveLabel; }

  /// Helper to invoke a method.
  bool ExecuteCall(Function *F, Pointer &&This, const SourceInfo &Info);
  /// Helper to emit a diagnostic on a missing method.
  bool ExecuteNoCall(const FunctionDecl *F, const SourceInfo &Info);

protected:
#define GET_EVAL_PROTO
#include "Opcodes.inc"
#undef GET_EVAL_PROTO
};

} // namespace interp
} // namespace clang

#endif
