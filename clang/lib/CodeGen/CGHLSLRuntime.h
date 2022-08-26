//===----- CGHLSLRuntime.h - Interface to HLSL Runtimes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for HLSL code generation.  Concrete
// subclasses of this implement code generation for specific HLSL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGHLSLRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGHLSLRUNTIME_H

#include "llvm/IR/IRBuilder.h"

#include "clang/Basic/HLSLRuntime.h"

namespace llvm {
class GlobalVariable;
class Function;
} // namespace llvm
namespace clang {
class VarDecl;
class ParmVarDecl;

class FunctionDecl;

namespace CodeGen {

class CodeGenModule;

class CGHLSLRuntime {
protected:
  CodeGenModule &CGM;
  uint32_t ResourceCounters[static_cast<uint32_t>(
      hlsl::ResourceClass::NumClasses)] = {0};

  llvm::Value *emitInputSemantic(llvm::IRBuilder<> &B, const ParmVarDecl &D);

public:
  CGHLSLRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGHLSLRuntime() {}

  void annotateHLSLResource(const VarDecl *D, llvm::GlobalVariable *GV);

  void finishCodeGen();

  void setHLSLEntryAttributes(const FunctionDecl *FD, llvm::Function *Fn);

  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);
};

} // namespace CodeGen
} // namespace clang

#endif
