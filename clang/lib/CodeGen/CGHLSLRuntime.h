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

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace llvm {
class GlobalVariable;
class Function;
class StructType;
} // namespace llvm

namespace clang {
class VarDecl;
class ParmVarDecl;
class HLSLBufferDecl;
class CallExpr;
class Type;
class DeclContext;

class FunctionDecl;

namespace CodeGen {

class CodeGenModule;

class CGHLSLRuntime {
public:
  struct Buffer {
    Buffer(const HLSLBufferDecl *D);
    llvm::StringRef Name;
    // IsCBuffer - Whether the buffer is a cbuffer (and not a tbuffer).
    bool IsCBuffer;
    llvm::Optional<unsigned> Reg;
    unsigned Space;
    // Global variable and offset for each constant.
    std::vector<std::pair<llvm::GlobalVariable *, unsigned>> Constants;
    llvm::StructType *LayoutStruct = nullptr;
  };

protected:
  CodeGenModule &CGM;
  uint32_t ResourceCounters[static_cast<uint32_t>(
      hlsl::ResourceClass::NumClasses)] = {0};

  llvm::Value *emitInputSemantic(llvm::IRBuilder<> &B, const ParmVarDecl &D);

public:
  CGHLSLRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGHLSLRuntime() {}

  void annotateHLSLResource(const VarDecl *D, llvm::GlobalVariable *GV);
  void generateGlobalCtorDtorCalls();

  void addBuffer(const HLSLBufferDecl *D);
  void finishCodeGen();

  void setHLSLEntryAttributes(const FunctionDecl *FD, llvm::Function *Fn);

  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);
  void setHLSLFunctionAttributes(llvm::Function *, const FunctionDecl *);

private:
  void addConstant(VarDecl *D, Buffer &CB);
  void addBufferDecls(const DeclContext *DC, Buffer &CB);
  llvm::SmallVector<Buffer> Buffers;
};

} // namespace CodeGen
} // namespace clang

#endif
