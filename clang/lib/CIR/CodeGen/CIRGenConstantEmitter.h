//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A helper class for emitting expressions and values as cir::ConstantOp
// and as initializers for global variables.
//
// Note: this is based on clang's LLVM IR codegen in ConstantEmitter.h, reusing
// this class interface makes it easier move forward with bringing CIR codegen
// to completion.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENCONSTANTEMITTER_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENCONSTANTEMITTER_H

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

namespace clang::CIRGen {

class ConstantEmitter {
public:
  CIRGenModule &cgm;
  const CIRGenFunction *cgf;

private:
  bool abstract = false;

#ifndef NDEBUG
  // Variables used for asserting state consistency.

  /// Whether non-abstract components of the emitter have been initialized.
  bool initializedNonAbstract = false;

  /// Whether the emitter has been finalized.
  bool finalized = false;

  /// Whether the constant-emission failed.
  bool failed = false;
#endif // NDEBUG

  /// Whether we're in a constant context.
  bool inConstantContext = false;

public:
  /// Initialize this emission in the context of the given function.
  /// Use this if the expression might contain contextual references like
  /// block addresses or PredefinedExprs.
  ConstantEmitter(CIRGenFunction &cgf) : cgm(cgf.cgm), cgf(&cgf) {}

  ConstantEmitter(CIRGenModule &cgm, CIRGenFunction *cgf = nullptr)
      : cgm(cgm), cgf(cgf) {}

  ConstantEmitter(const ConstantEmitter &other) = delete;
  ConstantEmitter &operator=(const ConstantEmitter &other) = delete;

  ~ConstantEmitter();

  /// Try to emit the initializer of the given declaration as an abstract
  /// constant.  If this succeeds, the emission must be finalized.
  mlir::Attribute tryEmitForInitializer(const VarDecl &d);

  void finalize(cir::GlobalOp gv);

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
  mlir::Attribute emitForMemory(mlir::Attribute c, QualType destType);

  /// Try to emit the initializer of the given declaration as an abstract
  /// constant.
  mlir::Attribute tryEmitAbstractForInitializer(const VarDecl &d);

  /// Emit the result of the given expression as an abstract constant,
  /// asserting that it succeeded.  This is only safe to do when the
  /// expression is known to be a constant expression with either a fairly
  /// simple type or a known simple form.
  mlir::Attribute emitAbstract(const Expr *e, QualType destType);
  mlir::Attribute emitAbstract(SourceLocation loc, const APValue &value,
                               QualType destType);

  mlir::Attribute tryEmitConstantExpr(const ConstantExpr *ce);

  // These are private helper routines of the constant emitter that
  // can't actually be private because things are split out into helper
  // functions and classes.

  mlir::Attribute tryEmitPrivateForVarInit(const VarDecl &d);

  mlir::TypedAttr tryEmitPrivate(const Expr *e, QualType destType);
  mlir::Attribute tryEmitPrivate(const APValue &value, QualType destType);
  mlir::Attribute tryEmitPrivateForMemory(const APValue &value, QualType t);

private:
#ifndef NDEBUG
  void initializeNonAbstract() {
    assert(!initializedNonAbstract);
    initializedNonAbstract = true;
    assert(!cir::MissingFeatures::addressSpace());
  }
  mlir::Attribute markIfFailed(mlir::Attribute init) {
    if (!init)
      failed = true;
    return init;
  }
#else
  void initializeNonAbstract() {}
  mlir::Attribute markIfFailed(mlir::Attribute init) { return init; }
#endif // NDEBUG

  class AbstractStateRAII {
    ConstantEmitter &emitter;
    bool oldValue;

  public:
    AbstractStateRAII(ConstantEmitter &emitter, bool value)
        : emitter(emitter), oldValue(emitter.abstract) {
      emitter.abstract = value;
    }
    ~AbstractStateRAII() { emitter.abstract = oldValue; }
  };
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENCONSTANTEMITTER_H
