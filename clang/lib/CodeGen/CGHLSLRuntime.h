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

#include "clang/Basic/Builtins.h"
#include "clang/Basic/HLSLRuntime.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"

#include <optional>
#include <vector>

#define GENERATE_HLSL_INTRINSIC_BASE(IntrinsicPostfix)                         \
  llvm::Triple::ArchType Arch = getArch();                                     \
  switch (Arch) {                                                              \
  case llvm::Triple::dxil:                                                     \
    return llvm::Intrinsic::dx_##IntrinsicPostfix;                             \
  case llvm::Triple::spirv:                                                    \
    return llvm::Intrinsic::spv_##IntrinsicPostfix;                            \
  default:                                                                     \
    llvm_unreachable("Intrinsic " #IntrinsicPostfix                            \
                     " not supported by target architecture");                 \
  }

// A function generator macro for when there is no builtins or
// when builtins are mapped to a set of intrinsics for different types.
#define GENERATE_HLSL_INTRINSIC_FUNCTION(FunctionName, IntrinsicPostfix)       \
  llvm::Intrinsic::ID get##FunctionName##Intrinsic() {                         \
    GENERATE_HLSL_INTRINSIC_BASE(IntrinsicPostfix)                             \
  }

// A template function generator macro for when we have builtins.
#define GENERATE_HLSL_INTRINSIC_TEMPLATE(BuiltinName, IntrinsicPostfix)        \
  template <> llvm::Intrinsic::ID getIntrinsic<Builtin::BI__##BuiltinName>() { \
    GENERATE_HLSL_INTRINSIC_BASE(IntrinsicPostfix)                             \
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

  template <unsigned BI> llvm::Intrinsic::ID getIntrinsic();
  GENERATE_HLSL_INTRINSIC_TEMPLATE(builtin_hlsl_elementwise_all, all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(ThreadId, thread_id)

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

// Default implementation
template <unsigned BI> llvm::Intrinsic::ID CGHLSLRuntime::getIntrinsic() {
  static_assert(false, "getIntrinsic is only allowed on specialized templates");
}

} // namespace CodeGen
} // namespace clang

#endif
