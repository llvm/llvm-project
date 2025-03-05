//===------ CIRGenCUDARuntime.h - Interface to CUDA Runtimes -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA CIR generation. Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
#define LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenModule;
class FunctionArgList;
class RValue;
class ReturnValueSlot;

class CIRGenCUDARuntime {
protected:
  CIRGenModule &cgm;
  StringRef Prefix;

  // Map a device stub function to a symbol for identifying kernel in host code.
  // For CUDA, the symbol for identifying the kernel is the same as the device
  // stub function. For HIP, they are different.
  llvm::DenseMap<StringRef, mlir::Operation *> KernelHandles;

  // Map a kernel handle to the kernel stub.
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> KernelStubs;

private:
  void emitDeviceStubBodyLegacy(CIRGenFunction &cgf, cir::FuncOp fn,
                                FunctionArgList &args);
  void emitDeviceStubBodyNew(CIRGenFunction &cgf, cir::FuncOp fn,
                             FunctionArgList &args);
  std::string addPrefixToName(StringRef FuncName) const;
  std::string addUnderscoredPrefixToName(StringRef FuncName) const;

public:
  CIRGenCUDARuntime(CIRGenModule &cgm);
  virtual ~CIRGenCUDARuntime();

  virtual void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                              FunctionArgList &args);

  virtual RValue emitCUDAKernelCallExpr(CIRGenFunction &cgf,
                                        const CUDAKernelCallExpr *expr,
                                        ReturnValueSlot retValue);
  virtual mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl GD);
  virtual void internalizeDeviceSideVar(const VarDecl *d,
                                        cir::GlobalLinkageKind &linkage);
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
