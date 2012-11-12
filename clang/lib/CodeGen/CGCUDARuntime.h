//===----- CGCUDARuntime.h - Interface to CUDA Runtimes ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA code generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CUDARUNTIME_H
#define CLANG_CODEGEN_CUDARUNTIME_H

namespace clang {

class CUDAKernelCallExpr;

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
  CGCUDARuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGCUDARuntime();

  virtual RValue EmitCUDAKernelCallExpr(CodeGenFunction &CGF,
                                        const CUDAKernelCallExpr *E,
                                        ReturnValueSlot ReturnValue);
  
  virtual void EmitDeviceStubBody(CodeGenFunction &CGF,
                                  FunctionArgList &Args) = 0;

};

/// Creates an instance of a CUDA runtime class.
CGCUDARuntime *CreateNVCUDARuntime(CodeGenModule &CGM);

}
}

#endif
