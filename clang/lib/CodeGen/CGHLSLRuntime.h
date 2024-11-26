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

#define GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(FunctionName,                 \
                                                 IntrinsicPostfix)             \
  GENERATE_HLSL_INTRINSIC_FUNCTION(FunctionName, IntrinsicPostfix, 1, 1)

// A function generator macro for picking the right intrinsic
// for the target backend
#define GENERATE_HLSL_INTRINSIC_FUNCTION(FunctionName, IntrinsicPostfix,       \
                                         IncludeDXIL, IncludeSPIRV)            \
  llvm::Intrinsic::ID get##FunctionName##Intrinsic() {                         \
    llvm::Triple::ArchType Arch = getArch();                                   \
    switch (Arch) {                                                            \
      /* Include DXIL case only if IncludeDXIL is true */                      \
      IF_INCLUDE(IncludeDXIL, case llvm::Triple::dxil                          \
                 : return llvm::Intrinsic::dx_##IntrinsicPostfix;)             \
      /* Include SPIRV case only if IncludeSPIRV is true */                    \
      IF_INCLUDE(IncludeSPIRV, case llvm::Triple::spirv                        \
                 : return llvm::Intrinsic::spv_##IntrinsicPostfix;)            \
                                                                               \
    default:                                                                   \
      llvm_unreachable("Intrinsic " #IntrinsicPostfix                          \
                       " not supported by target architecture");               \
    }                                                                          \
  }

#define IF_INCLUDE(Condition, Code) IF_INCLUDE_IMPL(Condition, Code)
#define IF_INCLUDE_IMPL(Condition, Code) IF_INCLUDE_##Condition(Code)

#define IF_INCLUDE_1(Code) Code
#define IF_INCLUDE_0(Code)

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

  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(All, all)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Any, any)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Cross, cross)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Degrees, degrees)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Frac, frac)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Length, length)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Lerp, lerp)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Normalize, normalize)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Rsqrt, rsqrt)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Saturate, saturate)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Sign, sign)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Step, step)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Radians, radians)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(ThreadId, thread_id)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(FDot, fdot)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(SDot, sdot)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(UDot, udot)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Dot4AddI8Packed, dot4add_i8packed)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(Dot4AddU8Packed, dot4add_u8packed)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(WaveActiveAnyTrue, wave_any)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(WaveActiveCountBits,
                                           wave_active_countbits)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(WaveIsFirstLane, wave_is_first_lane)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(WaveReadLaneAt, wave_readlane)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(FirstBitUHigh, firstbituhigh)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(FirstBitSHigh, firstbitshigh)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(NClamp, nclamp)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(SClamp, sclamp)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(UClamp, uclamp)

  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(CreateHandleFromBinding,
                                           handle_fromBinding)
  GENERATE_HLSL_INTRINSIC_FUNCTION_DEFAULT(BufferUpdateCounter,
                                           bufferUpdateCounter)

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

  llvm::Type *convertHLSLSpecificType(const Type *T);

  void annotateHLSLResource(const VarDecl *D, llvm::GlobalVariable *GV);
  void generateGlobalCtorDtorCalls();

  void addBuffer(const HLSLBufferDecl *D);
  void finishCodeGen();

  void setHLSLEntryAttributes(const FunctionDecl *FD, llvm::Function *Fn);

  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);
  void setHLSLFunctionAttributes(const FunctionDecl *FD, llvm::Function *Fn);
  void handleGlobalVarDefinition(const VarDecl *VD, llvm::GlobalVariable *Var);

  bool needsResourceBindingInitFn();
  llvm::Function *createResourceBindingInitFn();
  llvm::Instruction *getConvergenceToken(llvm::BasicBlock &BB);

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

  llvm::SmallVector<std::pair<const VarDecl *, llvm::GlobalVariable *>>
      ResourcesToBind;
};

} // namespace CodeGen
} // namespace clang

#endif
