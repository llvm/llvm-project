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

#include "llvm/ADT/DenseMap.h"
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

using ResourceClass = llvm::dxil::ResourceClass;

namespace llvm {
class GlobalVariable;
class Function;
class StructType;
class Metadata;
} // namespace llvm

namespace clang {
class NamedDecl;
class VarDecl;
class ParmVarDecl;
class InitListExpr;
class HLSLBufferDecl;
class HLSLVkBindingAttr;
class HLSLResourceBindingAttr;
class Type;
class RecordType;
class DeclContext;
class HLSLPackOffsetAttr;
class ArraySubscriptExpr;

class FunctionDecl;

namespace CodeGen {

class CodeGenModule;
class CodeGenFunction;
class LValue;

class CGHLSLRuntime {
public:
  //===----------------------------------------------------------------------===//
  // Start of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

  GENERATE_HLSL_INTRINSIC_FUNCTION(All, all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Any, any)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Cross, cross)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Degrees, degrees)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Frac, frac)
  GENERATE_HLSL_INTRINSIC_FUNCTION(FlattenedThreadIdInGroup,
                                   flattened_thread_id_in_group)
  GENERATE_HLSL_INTRINSIC_FUNCTION(IsInf, isinf)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Lerp, lerp)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Normalize, normalize)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Rsqrt, rsqrt)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Saturate, saturate)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Sign, sign)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Step, step)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Radians, radians)
  GENERATE_HLSL_INTRINSIC_FUNCTION(ThreadId, thread_id)
  GENERATE_HLSL_INTRINSIC_FUNCTION(GroupThreadId, thread_id_in_group)
  GENERATE_HLSL_INTRINSIC_FUNCTION(GroupId, group_id)
  GENERATE_HLSL_INTRINSIC_FUNCTION(FDot, fdot)
  GENERATE_HLSL_INTRINSIC_FUNCTION(SDot, sdot)
  GENERATE_HLSL_INTRINSIC_FUNCTION(UDot, udot)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Dot4AddI8Packed, dot4add_i8packed)
  GENERATE_HLSL_INTRINSIC_FUNCTION(Dot4AddU8Packed, dot4add_u8packed)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveActiveAllTrue, wave_all)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveActiveAnyTrue, wave_any)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveActiveCountBits, wave_active_countbits)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveIsFirstLane, wave_is_first_lane)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveGetLaneCount, wave_get_lane_count)
  GENERATE_HLSL_INTRINSIC_FUNCTION(WaveReadLaneAt, wave_readlane)
  GENERATE_HLSL_INTRINSIC_FUNCTION(FirstBitUHigh, firstbituhigh)
  GENERATE_HLSL_INTRINSIC_FUNCTION(FirstBitSHigh, firstbitshigh)
  GENERATE_HLSL_INTRINSIC_FUNCTION(FirstBitLow, firstbitlow)
  GENERATE_HLSL_INTRINSIC_FUNCTION(NClamp, nclamp)
  GENERATE_HLSL_INTRINSIC_FUNCTION(SClamp, sclamp)
  GENERATE_HLSL_INTRINSIC_FUNCTION(UClamp, uclamp)

  GENERATE_HLSL_INTRINSIC_FUNCTION(CreateResourceGetPointer,
                                   resource_getpointer)
  GENERATE_HLSL_INTRINSIC_FUNCTION(CreateHandleFromBinding,
                                   resource_handlefrombinding)
  GENERATE_HLSL_INTRINSIC_FUNCTION(CreateHandleFromImplicitBinding,
                                   resource_handlefromimplicitbinding)
  GENERATE_HLSL_INTRINSIC_FUNCTION(BufferUpdateCounter, resource_updatecounter)
  GENERATE_HLSL_INTRINSIC_FUNCTION(GroupMemoryBarrierWithGroupSync,
                                   group_memory_barrier_with_group_sync)

  //===----------------------------------------------------------------------===//
  // End of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

protected:
  CodeGenModule &CGM;

  llvm::Value *emitInputSemantic(llvm::IRBuilder<> &B, const ParmVarDecl &D,
                                 llvm::Type *Ty);

public:
  CGHLSLRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGHLSLRuntime() {}

  llvm::Type *
  convertHLSLSpecificType(const Type *T,
                          SmallVector<int32_t> *Packoffsets = nullptr);

  void generateGlobalCtorDtorCalls();

  void addBuffer(const HLSLBufferDecl *D);
  void finishCodeGen();

  void setHLSLEntryAttributes(const FunctionDecl *FD, llvm::Function *Fn);

  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);
  void setHLSLFunctionAttributes(const FunctionDecl *FD, llvm::Function *Fn);
  void handleGlobalVarDefinition(const VarDecl *VD, llvm::GlobalVariable *Var);

  llvm::Instruction *getConvergenceToken(llvm::BasicBlock &BB);

  llvm::TargetExtType *
  getHLSLBufferLayoutType(const RecordType *LayoutStructTy);
  void addHLSLBufferLayoutType(const RecordType *LayoutStructTy,
                               llvm::TargetExtType *LayoutTy);
  void emitInitListOpaqueValues(CodeGenFunction &CGF, InitListExpr *E);

  std::optional<LValue>
  emitResourceArraySubscriptExpr(const ArraySubscriptExpr *E,
                                 CodeGenFunction &CGF);

private:
  void emitBufferGlobalsAndMetadata(const HLSLBufferDecl *BufDecl,
                                    llvm::GlobalVariable *BufGV);
  void initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                   llvm::GlobalVariable *GV,
                                   HLSLVkBindingAttr *VkBinding);
  void initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                   llvm::GlobalVariable *GV,
                                   HLSLResourceBindingAttr *RBA);
  llvm::Triple::ArchType getArch();

  llvm::DenseMap<const clang::RecordType *, llvm::TargetExtType *> LayoutTypes;
};

} // namespace CodeGen
} // namespace clang

#endif
