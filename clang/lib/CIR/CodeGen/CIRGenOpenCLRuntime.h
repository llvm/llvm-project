//===-- CIRGenOpenCLRuntime.h - Interface to OpenCL Runtimes -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenCL CIR generation. Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENOPENCLRUNTIME_H
#define LLVM_CLANG_LIB_CIR_CIRGENOPENCLRUNTIME_H

namespace clang {

class VarDecl;

} // namespace clang

namespace cir {

class CIRGenFunction;
class CIRGenModule;

class CIRGenOpenCLRuntime {
protected:
  CIRGenModule &CGM;

public:
  CIRGenOpenCLRuntime(CIRGenModule &CGM) : CGM(CGM) {}
  virtual ~CIRGenOpenCLRuntime();

  /// Emit the IR required for a work-group-local variable declaration, and add
  /// an entry to CGF's LocalDeclMap for D.  The base class does this using
  /// CIRGenFunction::EmitStaticVarDecl to emit an internal global for D.
  virtual void buildWorkGroupLocalVarDecl(CIRGenFunction &CGF,
                                          const clang::VarDecl &D);
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CIRGENOPENCLRUNTIME_H
