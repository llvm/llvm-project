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
class NamedDecl;
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
  class DeviceVarFlags {
  public:
    enum DeviceVarKind {
      Variable, // Variable
      Surface,  // Builtin surface
      Texture,  // Builtin texture
    };

  private:
    unsigned Kind : 2;
    unsigned Extern : 1;
    unsigned Constant : 1;   // Constant variable.
    unsigned Managed : 1;    // Managed variable.
    unsigned Normalized : 1; // Normalized texture.
    int SurfTexType;         // Type of surface/texutre.

  public:
    DeviceVarFlags(DeviceVarKind K, bool E, bool C, bool M, bool N, int T)
        : Kind(K), Extern(E), Constant(C), Managed(M), Normalized(N),
          SurfTexType(T) {}

    DeviceVarKind getKind() const { return static_cast<DeviceVarKind>(Kind); }
    bool isExtern() const { return Extern; }
    bool isConstant() const { return Constant; }
    bool isManaged() const { return Managed; }
    bool isNormalized() const { return Normalized; }
    int getSurfTexType() const { return SurfTexType; }
  };

  CGCUDARuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGCUDARuntime();

  virtual RValue EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                        const CUDAKernelCallExpr *E,
                                        ReturnValueSlot ReturnValue);

  /// Emits a kernel launch stub.
  virtual void emitDeviceStub(CodeGenFunction &CGF, FunctionArgList &Args) = 0;
  virtual void registerDeviceVar(const VarDecl *VD, llvm::GlobalVariable &Var,
                                 bool Extern, bool Constant) = 0;
  virtual void registerDeviceSurf(const VarDecl *VD, llvm::GlobalVariable &Var,
                                  bool Extern, int Type) = 0;
  virtual void registerDeviceTex(const VarDecl *VD, llvm::GlobalVariable &Var,
                                 bool Extern, int Type, bool Normalized) = 0;

  /// Constructs and returns a module initialization function or nullptr if it's
  /// not needed. Must be called after all kernels have been emitted.
  virtual llvm::Function *makeModuleCtorFunction() = 0;

  /// Returns a module cleanup function or nullptr if it's not needed.
  /// Must be called after ModuleCtorFunction
  virtual llvm::Function *makeModuleDtorFunction() = 0;

  /// Returns function or variable name on device side even if the current
  /// compilation is for host.
  virtual std::string getDeviceSideName(const NamedDecl *ND) = 0;
};

/// Creates an instance of a CUDA runtime class.
CGCUDARuntime *CreateNVCUDARuntime(CodeGenModule &CGM);

}
}

#endif
