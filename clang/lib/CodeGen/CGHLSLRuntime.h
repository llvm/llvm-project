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

// A function generator macro for picking the right intrinsic
// for the target backend
#define GENERATE_HLSL_INTRINSIC_FUNCTION(FunctionName, IntrinsicPostfix)       \
  llvm::Intrinsic::ID get##FunctionName##Intrinsic() {                         \
    llvm::Triple::ArchType Arch = getArch();                                   \
    switch (Arch) {                                                            \
    case llvm::Triple::dxil:                                                   \
      return llvm::Intrinsic::dx_##IntrinsicPostfix;                           \
    case llvm::Triple::spirv:                                                  \
      return llvm::Intrinsic::spv_##IntrinsicPostfix;                          \
    default:                                                                   \
      llvm_unreachable("Intrinsic " #IntrinsicPostfix                          \
                       " not supported by target architecture");               \
    }                                                                          \
  }

namespace llvm {
class GlobalVariable;
class Function;
class StructType;
class Value;

template <> struct DenseMapInfo<clang::Builtin::ID> {
  static clang::Builtin::ID getEmptyKey() { return clang::Builtin::NotBuiltin; }

  static clang::Builtin::ID getTombstoneKey() {
    return clang::Builtin::FirstTSBuiltin;
  }

  static unsigned getHashValue(clang::Builtin::ID Val) {
    return static_cast<unsigned>(Val);
  }

  static bool isEqual(clang::Builtin::ID LHS, clang::Builtin::ID RHS) {
    return LHS == RHS;
  }
};

template <> struct DenseMapInfo<llvm::Triple::ArchType> {
  static llvm::Triple::ArchType getEmptyKey() {
    return llvm::Triple::ArchType::UnknownArch;
  }

  static llvm::Triple::ArchType getTombstoneKey() {
    return llvm::Triple::ArchType::LastArchType;
  }

  static unsigned getHashValue(llvm::Triple::ArchType Val) {
    return static_cast<unsigned>(Val);
  }

  static bool isEqual(llvm::Triple::ArchType LHS, llvm::Triple::ArchType RHS) {
    return LHS == RHS;
  }
};

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

struct CGHLSLIntrinsic {
  llvm::DenseMap<llvm::Triple::ArchType, llvm::function_ref<llvm::Value *()>>
      targetImplementations;
  llvm::function_ref<llvm::Value *()> genericImplementation =
      []() -> llvm::Value * {
    llvm_unreachable("Intrinsic not supported by target architecture.");
  };
};

class CGHLSLRuntime {
public:
  //===----------------------------------------------------------------------===//
  // Start of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

  GENERATE_HLSL_INTRINSIC_FUNCTION(All, all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Any, any)
  GENERATE_HLSL_INTRINSIC_FUNCTION(ThreadId, thread_id)

  //===----------------------------------------------------------------------===//
  // End of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//
  void registerHLSLTargetIntrinsic(Builtin::ID, llvm::Triple::ArchType,
                                   llvm::function_ref<llvm::Value *()>);
  void registerHLSLGenericIntrinsic(Builtin::ID,
                                    llvm::function_ref<llvm::Value *()>);
  llvm::Value *emitHLSLIntrinsic(Builtin::ID);
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
  llvm::DenseMap<clang::Builtin::ID, CGHLSLIntrinsic> IntrinsicCodeGen;
};

} // namespace CodeGen
} // namespace clang

#endif
