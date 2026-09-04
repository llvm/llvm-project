//===------ CIRGenCUDARuntime.h - Interface to CUDA Runtimes -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA code generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
#define LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace clang {
class CUDAKernelCallExpr;
}

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenModule;
class FunctionArgList;
class RValue;
class ReturnValueSlot;

class CIRGenCUDARuntime {
protected:
  CIRGenModule &cgm;

public:
  CIRGenCUDARuntime(CIRGenModule &cgm) : cgm(cgm) {}
  virtual ~CIRGenCUDARuntime();

  virtual void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                              FunctionArgList &args) = 0;

  virtual RValue emitCUDAKernelCallExpr(CIRGenFunction &cgf,
                                        const CUDAKernelCallExpr *expr,
                                        ReturnValueSlot retValue);

  virtual mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl gd) = 0;

  virtual mlir::Operation *getKernelStub(mlir::Operation *handle) = 0;

  /// Adjust linkage of shadow variables in host compilation
  virtual void internalizeDeviceSideVar(const VarDecl *d,
                                        cir::GlobalLinkageKind &linkage) = 0;

  /// Check whether a variable is a device variable and register it if true.
  virtual void handleVarRegistration(const VarDecl *vd, cir::GlobalOp var) = 0;

  /// Perform module finalization: on device side, mark ODR-used device
  /// variables as compiler-used. Mirrors OG's CGCUDARuntime::finalizeModule.
  virtual void finalizeModule() {}

  /// Returns function or variable name on device side even if the current
  /// compilation is for host.
  virtual std::string getDeviceSideName(const NamedDecl *nd) = 0;

  virtual void handleGlobalReplace(cir::GlobalOp oldGV, cir::GlobalOp newGV) {}
};

CIRGenCUDARuntime *createNVCUDARuntime(CIRGenModule &cgm);

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
