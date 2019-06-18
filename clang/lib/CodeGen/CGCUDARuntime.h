//===----- CGCUDARuntime.h - Interface to CUDA Runtimes ---------*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGCUDARUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGCUDARUNTIME_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class GlobalVariable;
}

namespace clang {

class CUDAKernelCallExpr;
class VarDecl;

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;
class FunctionArgList;
class ReturnValueSlot;
class RValue;

class CGCUDARuntime {
protected:
  CodeGenModule &CGM;

public:
  // Global variable properties that must be passed to CUDA runtime.
  enum DeviceVarFlags {
    ExternDeviceVar = 0x01,   // extern
    ConstantDeviceVar = 0x02, // __constant__
  };

  CGCUDARuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGCUDARuntime();

  virtual RValue EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                        const CUDAKernelCallExpr *E,
                                        ReturnValueSlot ReturnValue);

  /// Emits a kernel launch stub.
  virtual void emitDeviceStub(CodeGenFunction &CGF, FunctionArgList &Args) = 0;
  virtual void registerDeviceVar(const VarDecl *VD, llvm::GlobalVariable &Var,
                                 unsigned Flags) = 0;

  /// Constructs and returns a module initialization function or nullptr if it's
  /// not needed. Must be called after all kernels have been emitted.
  virtual llvm::Function *makeModuleCtorFunction() = 0;

  /// Returns a module cleanup function or nullptr if it's not needed.
  /// Must be called after ModuleCtorFunction
  virtual llvm::Function *makeModuleDtorFunction() = 0;

  /// Construct and return the stub name of a kernel.
  virtual std::string getDeviceStubName(llvm::StringRef Name) const = 0;
};

/// Creates an instance of a CUDA runtime class.
CGCUDARuntime *CreateNVCUDARuntime(CodeGenModule &CGM);

}
}

#endif
