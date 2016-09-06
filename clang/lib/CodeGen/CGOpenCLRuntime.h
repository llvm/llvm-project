//===----- CGOpenCLRuntime.h - Interface to OpenCL Runtimes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenCL code generation.  Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENCLRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENCLRUNTIME_H

#include "clang/AST/Type.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace clang {

class VarDecl;

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

class CGOpenCLRuntime {
protected:
  CodeGenModule &CGM;
  llvm::Type *PipeTy;
  llvm::PointerType *SamplerTy;

public:
  CGOpenCLRuntime(CodeGenModule &CGM) : CGM(CGM), PipeTy(nullptr),
    SamplerTy(nullptr) {}
  virtual ~CGOpenCLRuntime();

  /// Emit the IR required for a work-group-local variable declaration, and add
  /// an entry to CGF's LocalDeclMap for D.  The base class does this using
  /// CodeGenFunction::EmitStaticVarDecl to emit an internal global for D.
  virtual void EmitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                         const VarDecl &D);

  virtual llvm::Type *convertOpenCLSpecificType(const Type *T);

  virtual llvm::Type *getPipeType();

  llvm::PointerType *getSamplerType();
};

}
}

#endif
