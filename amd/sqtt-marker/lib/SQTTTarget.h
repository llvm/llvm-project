//===- SQTTTarget.h - SQTT target helpers ---------------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines AMDGPU target queries and instrumentation cost helpers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_AMD_SQTT_MARKER_LIB_SQTTTARGET_H
#define LLVM_AMD_SQTT_MARKER_LIB_SQTTTARGET_H

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#include "SQTTConfig.h"

using namespace llvm;

constexpr llvm::StringLiteral SqttMarkerHeaderMetadata = "sqtt.marker_header";
constexpr llvm::StringLiteral SqttRawPayloadMetadata = "sqtt.raw_payload";
constexpr llvm::StringLiteral SqttPayloadGroupMetadata = "sqtt.payload_group";

// ============================================================================
// Architecture detection
// ============================================================================

// This pass has no TargetMachine and must also build against installed LLVM
// releases where the public AMDGPU target-parser header has moved. Classify
// the small set of ISA protocol differences from the target-cpu attribute.
enum class GfxGen { GFX9, RDNA, GFX12, Unknown };

inline GfxGen getGfxGen(const llvm::Function &F) {
  llvm::Attribute A = F.getFnAttribute("target-cpu");
  if (!A.isValid())
    return GfxGen::Unknown;
  llvm::StringRef CPU = A.getValueAsString();
  if (CPU.starts_with("gfx9"))
    return GfxGen::GFX9;
  if (CPU.starts_with("gfx12"))
    return GfxGen::GFX12;
  if (CPU.starts_with("gfx10") || CPU.starts_with("gfx11"))
    return GfxGen::RDNA;
  return GfxGen::Unknown;
}

// gfx1200 and gfx1201 use HW_REG_SHADER_CYCLES_LO through s_getreg; gfx1250
// and subsequent targets have a dedicated shader-cycle instruction.
inline bool hasShaderCyclesU64(const llvm::Function &F) {
  llvm::Attribute A = F.getFnAttribute("target-cpu");
  if (!A.isValid())
    return false;
  llvm::StringRef CPU = A.getValueAsString();
  if (!CPU.consume_front("gfx"))
    return false;
  unsigned Target = 0;
  return !CPU.consumeInteger(10, Target) && Target > 1201;
}

// Does this GfxGen support s_ttracedata_imm?
inline bool supportsImmTrace(GfxGen Gen) {
  return Gen == GfxGen::RDNA || Gen == GfxGen::GFX12; // gfx10+
}

// s_nop N waits N+1 cycles; gfx10+ needs four cycles after an M0 write.
inline unsigned getM0TraceNop(GfxGen Gen) {
  return Gen == GfxGen::GFX9 ? 0 : 3;
}

// Wave size for this architecture
inline unsigned getWaveSize(GfxGen Gen) {
  return (Gen == GfxGen::GFX9) ? 64 : 32;
}

struct HwRegEncodings {
  uint32_t Wave, Simd, Cu, Wg;
};

inline HwRegEncodings getHwRegEncodings(GfxGen Gen) {
  if (Gen == GfxGen::GFX9)
    return {Gfx9HwregWave, Gfx9HwregSimd, Gfx9HwregCu, Gfx9HwregWg};
  return {RdnaHwregWave, RdnaHwregSimd, RdnaHwregCu, RdnaHwregWg};
}

inline llvm::Value *getMemoryPointer(llvm::Instruction *I) {
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I))
    return LI->getPointerOperand();
  if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(I))
    return SI->getPointerOperand();
  if (auto *AI = llvm::dyn_cast<llvm::AtomicRMWInst>(I))
    return AI->getPointerOperand();
  if (auto *AX = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(I))
    return AX->getPointerOperand();
  return nullptr;
}

// ============================================================================
// Instruction cost model
// ============================================================================

inline bool isCountedInstruction(const llvm::Instruction &I) {
  return !llvm::isa<llvm::PHINode>(I) && !llvm::isa<llvm::AllocaInst>(I) &&
         !llvm::isa<PseudoProbeInst>(I) && !llvm::isa<llvm::UnreachableInst>(I);
}

inline unsigned instructionCost(const llvm::Instruction &I) {
  if (!isCountedInstruction(I))
    return 0;
  // Check for lifetime intrinsics
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
    if (auto *F = CI->getCalledFunction()) {
      llvm::StringRef Name = F->getName();
      if (Name.starts_with("llvm.lifetime."))
        return 0;
      if (Name.starts_with("llvm.dbg."))
        return 0;
      // Matrix ops
      if (Name.starts_with("llvm.amdgcn.mfma.") ||
          Name.starts_with("llvm.amdgcn.wmma."))
        return 16;
      // LDS intrinsics
      if (Name.starts_with("llvm.amdgcn.ds."))
        return 4;
    }
  }
  // Memory operations
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
    unsigned AS = LI->getPointerAddressSpace();
    if (AS == 3)
      return 4; // LDS
    return 10;  // global/flat
  }
  if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
    unsigned AS = SI->getPointerAddressSpace();
    if (AS == 3)
      return 4;
    return 10;
  }
  return 1;
}

inline unsigned computeFunctionSize(const llvm::Function &F, CostMode Mode) {
  unsigned Total = 0;
  for (const llvm::BasicBlock &BB : F) {
    for (const llvm::Instruction &I : BB) {
      // Pass-owned marker calls must not make a function appear large
      // enough to retain the instrumentation that introduced them.
      if (I.getMetadata(SqttMarkerHeaderMetadata) ||
          I.getMetadata(SqttRawPayloadMetadata) ||
          I.getMetadata(SqttPayloadGroupMetadata))
        continue;
      Total += Mode == CostMode::WeightedCost ? instructionCost(I)
                                              : isCountedInstruction(I);
    }
  }
  return Total;
}

enum class BufferOpKind : uint8_t { None, Load, Store, Atomic };

inline BufferOpKind classifyBufferOp(llvm::StringRef Name) {
  for (const char *Prefix :
       {"llvm.amdgcn.raw.buffer.", "llvm.amdgcn.struct.buffer.",
        "llvm.amdgcn.raw.ptr.buffer.", "llvm.amdgcn.struct.ptr.buffer."}) {
    llvm::StringRef Opcode = Name;
    if (!Opcode.consume_front(Prefix))
      continue;
    if (Opcode.starts_with("load"))
      return BufferOpKind::Load;
    if (Opcode.starts_with("store"))
      return BufferOpKind::Store;
    if (Opcode.starts_with("atomic"))
      return BufferOpKind::Atomic;
    return BufferOpKind::None;
  }
  return BufferOpKind::None;
}

inline bool isStructBuffer(llvm::StringRef Name) {
  return Name.contains("struct");
}

inline bool isBufferCmpSwap(llvm::StringRef Name) {
  return Name.contains("cmpswap");
}

#endif // LLVM_AMD_SQTT_MARKER_LIB_SQTTTARGET_H
