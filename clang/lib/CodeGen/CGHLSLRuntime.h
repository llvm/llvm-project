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

#include "Address.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/HLSLRuntime.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

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
class HLSLRootSignatureDecl;
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

class CGHLSLOffsetInfo {
  SmallVector<uint32_t> Offsets;

public:
  static const uint32_t Unspecified = ~0U;

  /// Iterates over all declarations in the HLSL buffer and based on the
  /// packoffset or register(c#) annotations it fills outs the Offsets vector
  /// with the user-specified layout offsets. The buffer offsets can be
  /// specified 2 ways: 1. declarations in cbuffer {} block can have a
  /// packoffset annotation (translates to HLSLPackOffsetAttr) 2. default
  /// constant buffer declarations at global scope can have register(c#)
  /// annotations (translates to HLSLResourceBindingAttr with RegisterType::C)
  /// It is not guaranteed that all declarations in a buffer have an annotation.
  /// For those where it is not specified a `~0U` value is added to the Offsets
  /// vector. In the final layout these declarations will be placed at the end
  /// of the HLSL buffer after all of the elements with specified offset.
  static CGHLSLOffsetInfo fromDecl(const HLSLBufferDecl &BufDecl);

  /// Comparison function for offsets received from `operator[]` suitable for
  /// use in a `stable_sort`. This will order implicit bindings after explicit
  /// offsets.
  static bool compareOffsets(uint32_t LHS, uint32_t RHS) { return LHS < RHS; }

  /// Get the given offset, or `~0U` if there is no offset for the member.
  uint32_t operator[](size_t I) const {
    if (Offsets.empty())
      return Unspecified;
    return Offsets[I];
  }

  bool empty() const { return Offsets.empty(); }
};

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
  GENERATE_HLSL_INTRINSIC_FUNCTION(IsNaN, isnan)
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
  GENERATE_HLSL_INTRINSIC_FUNCTION(NonUniformResourceIndex,
                                   resource_nonuniformindex)
  GENERATE_HLSL_INTRINSIC_FUNCTION(BufferUpdateCounter, resource_updatecounter)
  GENERATE_HLSL_INTRINSIC_FUNCTION(GroupMemoryBarrierWithGroupSync,
                                   group_memory_barrier_with_group_sync)
  GENERATE_HLSL_INTRINSIC_FUNCTION(GetDimensionsX, resource_getdimensions_x)
  GENERATE_HLSL_INTRINSIC_FUNCTION(DdxCoarse, ddx_coarse)
  GENERATE_HLSL_INTRINSIC_FUNCTION(DdyCoarse, ddy_coarse)
  GENERATE_HLSL_INTRINSIC_FUNCTION(DdxFine, ddx_fine)
  GENERATE_HLSL_INTRINSIC_FUNCTION(DdyFine, ddy_fine)

  //===----------------------------------------------------------------------===//
  // End of reserved area for HLSL intrinsic getters.
  //===----------------------------------------------------------------------===//

protected:
  CodeGenModule &CGM;

  llvm::Value *emitSystemSemanticLoad(llvm::IRBuilder<> &B,
                                      const FunctionDecl *FD, llvm::Type *Type,
                                      const clang::DeclaratorDecl *Decl,
                                      HLSLAppliedSemanticAttr *Semantic,
                                      std::optional<unsigned> Index);

  void emitSystemSemanticStore(llvm::IRBuilder<> &B, llvm::Value *Source,
                               const clang::DeclaratorDecl *Decl,
                               HLSLAppliedSemanticAttr *Semantic,
                               std::optional<unsigned> Index);

  llvm::Value *handleScalarSemanticLoad(llvm::IRBuilder<> &B,
                                        const FunctionDecl *FD,
                                        llvm::Type *Type,
                                        const clang::DeclaratorDecl *Decl,
                                        HLSLAppliedSemanticAttr *Semantic);

  void handleScalarSemanticStore(llvm::IRBuilder<> &B, const FunctionDecl *FD,
                                 llvm::Value *Source,
                                 const clang::DeclaratorDecl *Decl,
                                 HLSLAppliedSemanticAttr *Semantic);

  std::pair<llvm::Value *, specific_attr_iterator<HLSLAppliedSemanticAttr>>
  handleStructSemanticLoad(
      llvm::IRBuilder<> &B, const FunctionDecl *FD, llvm::Type *Type,
      const clang::DeclaratorDecl *Decl,
      specific_attr_iterator<HLSLAppliedSemanticAttr> begin,
      specific_attr_iterator<HLSLAppliedSemanticAttr> end);

  specific_attr_iterator<HLSLAppliedSemanticAttr> handleStructSemanticStore(
      llvm::IRBuilder<> &B, const FunctionDecl *FD, llvm::Value *Source,
      const clang::DeclaratorDecl *Decl,
      specific_attr_iterator<HLSLAppliedSemanticAttr> AttrBegin,
      specific_attr_iterator<HLSLAppliedSemanticAttr> AttrEnd);

  std::pair<llvm::Value *, specific_attr_iterator<HLSLAppliedSemanticAttr>>
  handleSemanticLoad(llvm::IRBuilder<> &B, const FunctionDecl *FD,
                     llvm::Type *Type, const clang::DeclaratorDecl *Decl,
                     specific_attr_iterator<HLSLAppliedSemanticAttr> begin,
                     specific_attr_iterator<HLSLAppliedSemanticAttr> end);

  specific_attr_iterator<HLSLAppliedSemanticAttr>
  handleSemanticStore(llvm::IRBuilder<> &B, const FunctionDecl *FD,
                      llvm::Value *Source, const clang::DeclaratorDecl *Decl,
                      specific_attr_iterator<HLSLAppliedSemanticAttr> AttrBegin,
                      specific_attr_iterator<HLSLAppliedSemanticAttr> AttrEnd);

public:
  CGHLSLRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGHLSLRuntime() {}

  llvm::Type *convertHLSLSpecificType(const Type *T,
                                      const CGHLSLOffsetInfo &OffsetInfo);
  llvm::Type *convertHLSLSpecificType(const Type *T) {
    return convertHLSLSpecificType(T, CGHLSLOffsetInfo());
  }

  void generateGlobalCtorDtorCalls();

  void addBuffer(const HLSLBufferDecl *D);
  void addRootSignature(const HLSLRootSignatureDecl *D);
  void finishCodeGen();

  void setHLSLEntryAttributes(const FunctionDecl *FD, llvm::Function *Fn);

  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);
  void setHLSLFunctionAttributes(const FunctionDecl *FD, llvm::Function *Fn);
  void handleGlobalVarDefinition(const VarDecl *VD, llvm::GlobalVariable *Var);

  llvm::Instruction *getConvergenceToken(llvm::BasicBlock &BB);

  llvm::StructType *getHLSLBufferLayoutType(const RecordType *LayoutStructTy);
  void addHLSLBufferLayoutType(const RecordType *LayoutStructTy,
                               llvm::StructType *LayoutTy);
  void emitInitListOpaqueValues(CodeGenFunction &CGF, InitListExpr *E);

  std::optional<LValue>
  emitResourceArraySubscriptExpr(const ArraySubscriptExpr *E,
                                 CodeGenFunction &CGF);
  bool emitResourceArrayCopy(LValue &LHS, Expr *RHSExpr, CodeGenFunction &CGF);

  std::optional<LValue> emitBufferArraySubscriptExpr(
      const ArraySubscriptExpr *E, CodeGenFunction &CGF,
      llvm::function_ref<llvm::Value *(bool Promote)> EmitIdxAfterBase);

  bool emitBufferCopy(CodeGenFunction &CGF, Address DestPtr, Address SrcPtr,
                      QualType CType);

  LValue emitBufferMemberExpr(CodeGenFunction &CGF, const MemberExpr *E);

private:
  void emitBufferGlobalsAndMetadata(const HLSLBufferDecl *BufDecl,
                                    llvm::GlobalVariable *BufGV,
                                    const CGHLSLOffsetInfo &OffsetInfo);
  void initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                   llvm::GlobalVariable *GV);
  void initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                   llvm::GlobalVariable *GV,
                                   HLSLResourceBindingAttr *RBA);

  llvm::Value *emitSPIRVUserSemanticLoad(llvm::IRBuilder<> &B, llvm::Type *Type,
                                         const clang::DeclaratorDecl *Decl,
                                         HLSLAppliedSemanticAttr *Semantic,
                                         std::optional<unsigned> Index);
  llvm::Value *emitDXILUserSemanticLoad(llvm::IRBuilder<> &B, llvm::Type *Type,
                                        HLSLAppliedSemanticAttr *Semantic,
                                        std::optional<unsigned> Index);
  llvm::Value *emitUserSemanticLoad(llvm::IRBuilder<> &B, llvm::Type *Type,
                                    const clang::DeclaratorDecl *Decl,
                                    HLSLAppliedSemanticAttr *Semantic,
                                    std::optional<unsigned> Index);

  void emitSPIRVUserSemanticStore(llvm::IRBuilder<> &B, llvm::Value *Source,
                                  const clang::DeclaratorDecl *Decl,
                                  HLSLAppliedSemanticAttr *Semantic,
                                  std::optional<unsigned> Index);
  void emitDXILUserSemanticStore(llvm::IRBuilder<> &B, llvm::Value *Source,
                                 HLSLAppliedSemanticAttr *Semantic,
                                 std::optional<unsigned> Index);
  void emitUserSemanticStore(llvm::IRBuilder<> &B, llvm::Value *Source,
                             const clang::DeclaratorDecl *Decl,
                             HLSLAppliedSemanticAttr *Semantic,
                             std::optional<unsigned> Index);

  llvm::Triple::ArchType getArch();

  llvm::DenseMap<const clang::RecordType *, llvm::StructType *> LayoutTypes;
  unsigned SPIRVLastAssignedInputSemanticLocation = 0;
  unsigned SPIRVLastAssignedOutputSemanticLocation = 0;
};

} // namespace CodeGen
} // namespace clang

#endif
