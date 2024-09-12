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

#include "CIRGenBuilder.h"
#include "CIRGenValue.h"

#include "clang/AST/Redeclarable.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "llvm/Support/ErrorHandling.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"

#include "clang/CIR/MissingFeatures.h"

namespace clang {
class Decl;
class Expr;
class GlobalDecl;
class VarDecl;
} // namespace clang

namespace cir {
class CIRGenModule;
class CIRGenFunction;

struct OMPTaskDataTy final {
  struct DependData {
    clang::OpenMPDependClauseKind DepKind = clang::OMPC_DEPEND_unknown;
    const clang::Expr *IteratorExpr = nullptr;
    llvm::SmallVector<const clang::Expr *, 4> DepExprs;
    explicit DependData() = default;
    DependData(clang::OpenMPDependClauseKind DepKind,
               const clang::Expr *IteratorExpr)
        : DepKind(DepKind), IteratorExpr(IteratorExpr) {}
  };
  llvm::SmallVector<DependData, 4> Dependences;
  bool HasNowaitClause = false;
};

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

  /// Emit code for 'taskwait' directive
  virtual void emitTaskWaitCall(CIRGenBuilderTy &builder, CIRGenFunction &CGF,
                                mlir::Location Loc, const OMPTaskDataTy &Data);

  virtual void emitBarrierCall(CIRGenBuilderTy &builder, CIRGenFunction &CGF,
                               mlir::Location Loc);

  virtual void emitTaskyieldCall(CIRGenBuilderTy &builder, CIRGenFunction &CGF,
                                 mlir::Location Loc);

protected:
  CIRGenModule &CGM;
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
