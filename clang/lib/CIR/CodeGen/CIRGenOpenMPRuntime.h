//===--- CIRGenOpenMPRuntime.h - Interface to OpenMP Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime MLIR code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H

#include "CIRGenValue.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace clang {
class Decl;
class Expr;
class GlobalDecl;
class VarDecl;
} // namespace clang

namespace cir {
class CIRGenModule;
class CIRGenFunction;

class CIRGenOpenMPRuntime {
public:
  explicit CIRGenOpenMPRuntime(CIRGenModule &CGM);
  virtual ~CIRGenOpenMPRuntime() {}

  /// Gets the OpenMP-specific address of the local variable.
  virtual Address getAddressOfLocalVariable(CIRGenFunction &CGF,
                                            const clang::VarDecl *VD);

  /// Checks if the provided \p LVal is lastprivate conditional and emits the
  /// code to update the value of the original variable.
  /// \code
  /// lastprivate(conditional: a)
  /// ...
  /// <type> a;
  /// lp_a = ...;
  /// #pragma omp critical(a)
  /// if (last_iv_a <= iv) {
  ///   last_iv_a = iv;
  ///   global_a = lp_a;
  /// }
  /// \endcode
  virtual void checkAndEmitLastprivateConditional(CIRGenFunction &CGF,
                                                  const clang::Expr *LHS);

  /// Checks if the provided global decl \a GD is a declare target variable and
  /// registers it when emitting code for the host.
  virtual void registerTargetGlobalVariable(const clang::VarDecl *VD,
                                            mlir::cir::GlobalOp globalOp);

  /// Emit deferred declare target variables marked for deferred emission.
  void emitDeferredTargetDecls() const;

  /// Emits OpenMP-specific function prolog.
  /// Required for device constructs.
  virtual void emitFunctionProlog(CIRGenFunction &CGF, const clang::Decl *D);

  /// Emit the global \a GD if it is meaningful for the target. Returns
  /// if it was emitted successfully.
  /// \param GD Global to scan.
  virtual bool emitTargetGlobal(clang::GlobalDecl &D);

protected:
  CIRGenModule &CGM;
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
