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
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#include "clang/Basic/HLSLRuntime.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"

#include <optional>
#include <vector>

// Define the function generator macro
#define GENERATE_HLSL_INTRINSIC_FUNCTION(name)                                 \
  llvm::Intrinsic::ID get_hlsl_##name##_intrinsic() {                          \
    llvm::Triple::ArchType Arch = getArch();                                   \
    switch (Arch) {                                                            \
    case llvm::Triple::dxil:                                                   \
      return llvm::Intrinsic::dx_##name;                                       \
    case llvm::Triple::spirv:                                                  \
      return llvm::Intrinsic::spv_##name;                                      \
    default:                                                                   \
      llvm_unreachable("Intrinsic " #name                                      \
                       " not supported by target architecture");               \
    }                                                                          \
  }

namespace llvm {
class GlobalVariable;
class Function;
class StructType;
} // namespace llvm

namespace clang {
class VarDecl;
class ParmVarDecl;
class HLSLBufferDecl;
class HLSLResourceBindingAttr;
class Type;
class DeclContext;

class FunctionDecl;

namespace CodeGen {

class CodeGenModule;

class CGHLSLRuntime {
public:
  //===----------------------------------------------------------------------===//
  // Start of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

  GENERATE_HLSL_INTRINSIC_FUNCTION(all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(thread_id)

  //===----------------------------------------------------------------------===//
  // End of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

  struct BufferResBinding {
    // The ID like 2 in register(b2, space1).
    std::optional<unsigned> Reg;
    // The Space like 1 is register(b2, space1).
    // Default value is 0.
    unsigned Space;
    BufferResBinding(HLSLResourceBindingAttr *Attr);
  };
  struct Buffer {
    Buffer(const HLSLBufferDecl *D);
    llvm::StringRef Name;
    // IsCBuffer - Whether the buffer is a cbuffer (and not a tbuffer).
    bool IsCBuffer;
    BufferResBinding Binding;
    // Global variable and offset for each constant.
    std::vector<std::pair<llvm::GlobalVariable *, unsigned>> Constants;
    llvm::StructType *LayoutStruct = nullptr;
  };

protected:
  CodeGenModule &CGM;

  llvm::Value *emitInputSemantic(llvm::IRBuilder<> &B, const ParmVarDecl &D,
                                 llvm::Type *Ty);

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
  void addBufferResourceAnnotation(llvm::GlobalVariable *GV,
                                   llvm::hlsl::ResourceClass RC,
                                   llvm::hlsl::ResourceKind RK, bool IsROV,
                                   llvm::hlsl::ElementType ET,
                                   BufferResBinding &Binding);
  void addConstant(VarDecl *D, Buffer &CB);
  void addBufferDecls(const DeclContext *DC, Buffer &CB);
  llvm::Triple::ArchType getArch();
  llvm::SmallVector<Buffer> Buffers;
};

} // namespace CodeGen
} // namespace clang

#endif
