//===--- CIRGenCstEmitter.h - CIR constant emission -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A helper class for emitting expressions and values as mlir::cir::ConstantOp
// and as initializers for global variables.
//
// Note: this is based on LLVM's codegen in ConstantEmitter.h, reusing this
// class interface makes it easier move forward with bringing CIR codegen
// to completion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGEN_CONSTANTEMITTER_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGEN_CONSTANTEMITTER_H

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

namespace cir {

class ConstantEmitter {
public:
  CIRGenModule &CGM;
  CIRGenFunction *const CGF;

private:
  bool Abstract = false;

  /// Whether non-abstract components of the emitter have been initialized.
  bool InitializedNonAbstract = false;

  /// Whether the emitter has been finalized.
  bool Finalized = false;

  /// Whether the constant-emission failed.
  bool Failed = false;

  /// Whether we're in a constant context.
  bool InConstantContext = false;

  /// The AST address space where this (non-abstract) initializer is going.
  /// Used for generating appropriate placeholders.
  clang::LangAS DestAddressSpace;

  llvm::SmallVector<std::pair<llvm::Constant *, llvm::GlobalVariable *>, 4>
      PlaceholderAddresses;

public:
  ConstantEmitter(CIRGenModule &CGM, CIRGenFunction *CGF = nullptr)
      : CGM(CGM), CGF(CGF) {}

  /// Initialize this emission in the context of the given function.
  /// Use this if the expression might contain contextual references like
  /// block addresses or PredefinedExprs.
  ConstantEmitter(CIRGenFunction &CGF) : CGM(CGF.CGM), CGF(&CGF) {}

  ConstantEmitter(const ConstantEmitter &other) = delete;
  ConstantEmitter &operator=(const ConstantEmitter &other) = delete;

  ~ConstantEmitter();

  /// Is the current emission context abstract?
  bool isAbstract() const { return Abstract; }

  bool isInConstantContext() const { return InConstantContext; }
  void setInConstantContext(bool var) { InConstantContext = var; }

  /// Try to emit the initiaizer of the given declaration as an abstract
  /// constant.  If this succeeds, the emission must be finalized.
  mlir::Attribute tryEmitForInitializer(const VarDecl &D);
  mlir::Attribute tryEmitForInitializer(const Expr *E, LangAS destAddrSpace,
                                        QualType destType);

  void finalize(mlir::cir::GlobalOp global);

  // All of the "abstract" emission methods below permit the emission to
  // be immediately discarded without finalizing anything.  Therefore, they
  // must also promise not to do anything that will, in the future, require
  // finalization:
  //
  //   - using the CGF (if present) for anything other than establishing
  //     semantic context; for example, an expression with ignored
  //     side-effects must not be emitted as an abstract expression
  //
  //   - doing anything that would not be safe to duplicate within an
  //     initializer or to propagate to another context; for example,
  //     side effects, or emitting an initialization that requires a
  //     reference to its current location.
  mlir::Attribute emitForMemory(mlir::Attribute C, QualType T) {
    return emitForMemory(CGM, C, T);
  }

  // static llvm::Constant *emitNullForMemory(CodeGenModule &CGM, QualType T);
  static mlir::Attribute emitForMemory(CIRGenModule &CGM, mlir::Attribute C,
                                       clang::QualType T);

  /// Try to emit the initializer of the given declaration as an abstract
  /// constant.
  mlir::Attribute tryEmitAbstractForInitializer(const VarDecl &D);

  /// Emit the result of the given expression as an abstract constant,
  /// asserting that it succeeded.  This is only safe to do when the
  /// expression is known to be a constant expression with either a fairly
  /// simple type or a known simple form.
  mlir::Attribute emitAbstract(const Expr *E, QualType T);
  mlir::Attribute emitAbstract(SourceLocation loc, const APValue &value,
                               QualType T);

  mlir::Attribute tryEmitConstantExpr(const ConstantExpr *CE);

  // These are private helper routines of the constant emitter that
  // can't actually be private because things are split out into helper
  // functions and classes.

  mlir::Attribute tryEmitPrivateForVarInit(const VarDecl &D);
  mlir::TypedAttr tryEmitPrivate(const Expr *E, QualType T);
  mlir::TypedAttr tryEmitPrivateForMemory(const Expr *E, QualType T);

  mlir::Attribute tryEmitPrivate(const APValue &value, QualType T);
  mlir::Attribute tryEmitPrivateForMemory(const APValue &value, QualType T);

  mlir::Attribute tryEmitAbstract(const Expr *E, QualType destType);
  mlir::Attribute tryEmitAbstractForMemory(const Expr *E, QualType destType);

  mlir::Attribute tryEmitAbstract(const APValue &value, QualType destType);
  mlir::Attribute tryEmitAbstractForMemory(const APValue &value,
                                           QualType destType);

private:
  void initializeNonAbstract(clang::LangAS destAS) {
    assert(!InitializedNonAbstract);
    InitializedNonAbstract = true;
    DestAddressSpace = destAS;
  }
  mlir::Attribute markIfFailed(mlir::Attribute init) {
    if (!init)
      Failed = true;
    return init;
  }

  struct AbstractState {
    bool OldValue;
    size_t OldPlaceholdersSize;
  };
  AbstractState pushAbstract() {
    AbstractState saved = {Abstract, PlaceholderAddresses.size()};
    Abstract = true;
    return saved;
  }
  mlir::Attribute validateAndPopAbstract(mlir::Attribute C, AbstractState save);
};

} // namespace cir

#endif
