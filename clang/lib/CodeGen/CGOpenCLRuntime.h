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

#ifndef CLANG_CODEGEN_OPENCLRUNTIME_H
#define CLANG_CODEGEN_OPENCLRUNTIME_H

namespace clang {

class VarDecl;

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

class CGOpenCLRuntime {
protected:
  CodeGenModule &CGM;

public:
  CGOpenCLRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGOpenCLRuntime();

  /// Emit the IR required for a work-group-local variable declaration, and add
  /// an entry to CGF's LocalDeclMap for D.  The base class does this using
  /// CodeGenFunction::EmitStaticVarDecl to emit an internal global for D.
  virtual void EmitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                         const VarDecl &D);
};

}
}

#endif
