//===--- Context.h - Context for the constexpr VM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the constexpr execution context.
//
// The execution context manages cached bytecode and the global context.
// It invokes the compiler and interpreter, propagating errors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_CONTEXT_H
#define LLVM_CLANG_AST_INTERP_CONTEXT_H

#include "InterpStack.h"

namespace clang {
class ASTContext;
class LangOptions;
class FunctionDecl;
class VarDecl;
class APValue;
class BlockExpr;

namespace interp {
class Function;
class Program;
class State;
enum PrimType : unsigned;

struct ParamOffset {
  unsigned Offset;
  bool IsPtr;
};

/// Holds all information required to evaluate constexpr code in a module.
class Context final {
public:
  /// Initialises the constexpr VM.
  Context(ASTContext &Ctx);

  /// Cleans up the constexpr VM.
  ~Context();

  /// Checks if a function is a potential constant expression.
  bool isPotentialConstantExpr(State &Parent, const FunctionDecl *FD);
  void isPotentialConstantExprUnevaluated(State &Parent, const Expr *E,
                                          const FunctionDecl *FD);

  /// Evaluates a toplevel expression as an rvalue.
  bool evaluateAsRValue(State &Parent, const Expr *E, APValue &Result);

  /// Like evaluateAsRvalue(), but does no implicit lvalue-to-rvalue conversion.
  bool evaluate(State &Parent, const Expr *E, APValue &Result,
                ConstantExprKind Kind);

  /// Evaluates a toplevel initializer.
  bool evaluateAsInitializer(State &Parent, const VarDecl *VD, APValue &Result);

  bool evaluateCharRange(State &Parent, const Expr *SizeExpr,
                         const Expr *PtrExpr, APValue &Result);
  bool evaluateCharRange(State &Parent, const Expr *SizeExpr,
                         const Expr *PtrExpr, std::string &Result);

  /// Evalute \param E and if it can be evaluated to a string literal,
  /// run strlen() on it.
  bool evaluateStrlen(State &Parent, const Expr *E, uint64_t &Result);

  /// Returns the AST context.
  ASTContext &getASTContext() const { return Ctx; }
  /// Returns the language options.
  const LangOptions &getLangOpts() const;
  /// Returns CHAR_BIT.
  unsigned getCharBit() const;
  /// Return the floating-point semantics for T.
  const llvm::fltSemantics &getFloatSemantics(QualType T) const;
  /// Return the size of T in bits.
  uint32_t getBitWidth(QualType T) const { return Ctx.getIntWidth(T); }

  /// Classifies a type.
  OptPrimType classify(QualType T) const;

  /// Classifies an expression.
  OptPrimType classify(const Expr *E) const {
    assert(E);
    if (E->isGLValue())
      return PT_Ptr;

    return classify(E->getType());
  }

  const CXXMethodDecl *
  getOverridingFunction(const CXXRecordDecl *DynamicDecl,
                        const CXXRecordDecl *StaticDecl,
                        const CXXMethodDecl *InitialFunction) const;

  const Function *getOrCreateFunction(const FunctionDecl *FuncDecl);
  const Function *getOrCreateObjCBlock(const BlockExpr *E);

  /// Returns whether we should create a global variable for the
  /// given ValueDecl.
  static bool shouldBeGloballyIndexed(const ValueDecl *VD) {
    if (const auto *V = dyn_cast<VarDecl>(VD))
      return V->hasGlobalStorage() || V->isConstexpr();

    return false;
  }

  /// Returns the program. This is only needed for unittests.
  Program &getProgram() const { return *P; }

  unsigned collectBaseOffset(const RecordDecl *BaseDecl,
                             const RecordDecl *DerivedDecl) const;

  const Record *getRecord(const RecordDecl *D) const;

  unsigned getEvalID() const { return EvalID; }

  /// Unevaluated builtins don't get their arguments put on the stack
  /// automatically. They instead operate on the AST of their Call
  /// Expression.
  /// Similar information is available via ASTContext::BuiltinInfo,
  /// but that is not correct for our use cases.
  static bool isUnevaluatedBuiltin(unsigned ID);

private:
  /// Runs a function.
  bool Run(State &Parent, const Function *Func);

  template <typename ResultT>
  bool evaluateStringRepr(State &Parent, const Expr *SizeExpr,
                          const Expr *PtrExpr, ResultT &Result);

  /// Current compilation context.
  ASTContext &Ctx;
  /// Interpreter stack, shared across invocations.
  InterpStack Stk;
  /// Constexpr program.
  std::unique_ptr<Program> P;
  /// ID identifying an evaluation.
  unsigned EvalID = 0;
  /// Cached widths (in bits) of common types, for a faster classify().
  unsigned ShortWidth;
  unsigned IntWidth;
  unsigned LongWidth;
  unsigned LongLongWidth;
};

} // namespace interp
} // namespace clang

#endif
