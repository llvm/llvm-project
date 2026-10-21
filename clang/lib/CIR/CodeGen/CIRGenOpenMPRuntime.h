//===--- CIRGenOpenMPRuntime.h - OpenMP code generation helpers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H

#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseSet.h"

namespace clang::CIRGen {

class CIRGenModule;

class CIRGenOpenMPRuntime {
  CIRGenModule &cgm;

  /// Declarations that have been force-emitted for the target device because
  /// they are transitively referenced from declare target functions.
  llvm::DenseSet<CanonicalDeclPtr<const Decl>> alreadyEmittedTargetDecls;

  /// Returns false if the given function or declare reduction should be
  /// emitted. Returns true if it should eb skipped.
  /// emission).
  bool emitTargetFunctions(GlobalDecl gd);

  /// Returns false if given global variable should be emitted. Returns
  /// true if it should be skipped.
  bool emitTargetGlobalVariable(GlobalDecl gd);

public:
  explicit CIRGenOpenMPRuntime(CIRGenModule &cgm) : cgm(cgm) {}

  /// Check whether the given GlobalDecl needs special handling for device
  /// compilation. Returns false if it should be emitted, true if it should be
  /// skipped.
  bool emitTargetGlobal(GlobalDecl gd);

  /// Mark a function reference as one that should be emitted on the device.
  /// Returns false if it should be emitted, true if the function is already
  /// handled and should be skipped.
  bool markAsGlobalTarget(GlobalDecl gd);

  /// If the function has an OMPDeclareTargetDeclAttr, set the corresponding
  /// omp.declare_target attribute on the emitted cir.func op.
  void emitDeclareTargetFunction(const FunctionDecl *fd, cir::FuncOp funcOp);
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPRUNTIME_H
