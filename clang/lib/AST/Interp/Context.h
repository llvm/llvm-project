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

#include "Context.h"
#include "InterpStack.h"
#include "clang/AST/APValue.h"
#include "llvm/ADT/PointerIntPair.h"

namespace clang {
class ASTContext;
class LangOptions;
class Stmt;
class FunctionDecl;
class VarDecl;

namespace interp {
class Function;
class Program;
class State;
enum PrimType : unsigned;

/// Wrapper around interpreter termination results.
enum class InterpResult {
  /// Interpreter successfully computed a value.
  Success,
  /// Interpreter encountered an error and quit.
  Fail,
  /// Interpreter encountered an unimplemented feature, AST fallback.
  Bail,
};

/// Holds all information required to evaluate constexpr code in a module.
class Context {
public:
  /// Initialises the constexpr VM.
  Context(ASTContext &Ctx);

  /// Cleans up the constexpr VM.
  ~Context();

  /// Checks if a function is a potential constant expression.
  InterpResult isPotentialConstantExpr(State &Parent,
                                       const FunctionDecl *FnDecl);

  /// Evaluates a toplevel expression as an rvalue.
  InterpResult evaluateAsRValue(State &Parent, const Expr *E, APValue &Result);

  /// Evaluates a toplevel initializer.
  InterpResult evaluateAsInitializer(State &Parent, const VarDecl *VD,
                                     APValue &Result);

  /// Returns the AST context.
  ASTContext &getASTContext() const { return Ctx; }
  /// Returns the language options.
  const LangOptions &getLangOpts() const;
  /// Returns the interpreter stack.
  InterpStack &getStack() { return Stk; }
  /// Returns CHAR_BIT.
  unsigned getCharBit() const;

  /// Classifies an expression.
  llvm::Optional<PrimType> classify(QualType T);

private:
  /// Runs a function.
  InterpResult Run(State &Parent, Function *Func, APValue &Result);

  /// Checks a result fromt the interpreter.
  InterpResult Check(State &Parent, llvm::Expected<bool> &&R);

private:
  /// Current compilation context.
  ASTContext &Ctx;
  /// Flag to indicate if the use of the interpreter is mandatory.
  bool ForceInterp;
  /// Interpreter stack, shared across invocations.
  InterpStack Stk;
  /// Constexpr program.
  std::unique_ptr<Program> P;
};

} // namespace interp
} // namespace clang

#endif
