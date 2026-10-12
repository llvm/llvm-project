//===- RaiseContextTest.cpp - raise context unit tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/raise-context.h"

#include "transpiler/common/kernel-meta.h"
#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/raise_failure.h"
#include "transpiler/raiser/wave-projection.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace llvm;
using namespace COMGR::transpiler;

namespace {

class RaiseContextTest : public ::testing::Test {
protected:
  // Offset the source kernel starts at. Deliberately not zero: the mapping
  // tracks the kernel's own start, not the start of the text section it sits
  // in.
  static constexpr uint64_t KKernelStartOffset = 0x40;

  void SetUp() override {
    Expected<MCState> State = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
    Mc = std::move(*State);
    Env = std::make_unique<ContextEnvironment>(Mc);
  }

  struct ContextEnvironment {
    LLVMContext LLVMCtx;
    Module Mod;
    IRBuilder<> B;
    ReplicationProjection Projection;
    Function *Kernel;
    BasicBlock *Entry;
    SetPcAnalysis SetPc;
    DenseSet<uint64_t> PairedSplitBarriers;
    std::optional<RaiseContext> Ctx;

    explicit ContextEnvironment(const MCState &Mc)
        : Mod("raise_context_test", LLVMCtx), B(LLVMCtx),
          Projection(*Mc.SubtargetInfo, *Mc.SubtargetInfo, B.getInt32Ty(),
                     B.getInt64Ty()),
          Kernel(Function::Create(FunctionType::get(B.getVoidTy(),
                                                    {B.getInt32Ty()},
                                                    /*isVarArg=*/false),
                                  Function::ExternalLinkage, "kernel", Mod)),
          Entry(BasicBlock::Create(LLVMCtx, "entry", Kernel)) {
      B.SetInsertPoint(Entry);
      Ctx.emplace(cantFail(RaiseContext::create(
          B, Projection, Mc, SetPc, PairedSplitBarriers, KernelMeta(),
          ArrayRef<uint8_t>(), 0, ArrayRef<TextSection::ImageSection>(),
          KKernelStartOffset, 0)));
    }
  };

  MCState Mc;
  std::unique_ptr<ContextEnvironment> Env;
};

TEST_F(RaiseContextTest, ResolvesBlocksBySourceOffset) {
  BasicBlock *Start = BasicBlock::Create(Env->LLVMCtx, "bb_start", Env->Kernel);
  Env->Ctx->defineBB(KKernelStartOffset, Start);
  EXPECT_EQ(Env->Ctx->lookupBB(KKernelStartOffset), Start);
}

TEST_F(RaiseContextTest, SourceSramEcc) {
  for (StringRef Cpu : {"gfx1250", "gfx942"}) {
    Expected<MCState> State = initMCState(Cpu);
    ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
    for (std::optional<bool> Setting :
         {std::optional<bool>(), std::optional<bool>(false),
          std::optional<bool>(true)}) {
      ContextEnvironment Context(*State);
      Expected<RaiseContext> Result = RaiseContext::create(
          Context.B, Context.Projection, *State, Context.SetPc,
          Context.PairedSplitBarriers, KernelMeta(), {}, 0, {},
          KKernelStartOffset, 0, Setting);
      if (Cpu == "gfx1250" && Setting == false) {
        ASSERT_FALSE(static_cast<bool>(Result));
        handleAllErrors(Result.takeError(), [](const RaiseFailure &Failure) {
          EXPECT_EQ(Failure.reason(), RaiseFailureReason::BadInput);
          EXPECT_EQ(Failure.detail(),
                    "gfx1250 source does not support disabling SRAM ECC");
        });
        continue;
      }
      ASSERT_TRUE(static_cast<bool>(Result)) << toString(Result.takeError());
      EXPECT_EQ(Result->sourceSramEcc(),
                Cpu == "gfx1250" ? std::optional<bool>(true) : Setting);
    }
  }
}

TEST_F(RaiseContextTest, RequiredBitsFollowRegisterPromotion) {
  DecodedInst Instruction;
  AllocaInst *Word = Env->B.CreateAlloca(Env->B.getInt32Ty());
  Env->B.CreateStore(Env->B.getInt32(63), Word);
  Value *Read = Env->B.CreateLoad(Env->B.getInt32Ty(), Word);
  Env->Ctx->requireZeroBits(Read, 0xffffffc0, Instruction, "nonzero bits");
  Env->B.CreateRetVoid();
  DominatorTree Dominators(*Env->Kernel);
  PromoteMemToReg({Word}, Dominators);
  if (Error Result = Env->Ctx->validateRequiredBits())
    FAIL() << toString(std::move(Result));
}

TEST_F(RaiseContextTest, BufferRejectsMalformedOperands) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  unsigned Opcode = State->InstrInfo->getNumOpcodes();
  for (unsigned I = 0; I != State->InstrInfo->getNumOpcodes(); ++I) {
    if (State->InstrInfo->getName(I) ==
        "BUFFER_LOAD_DWORD_VBUFFER_OFFSET_gfx12") {
      Opcode = I;
      break;
    }
  }
  ASSERT_NE(Opcode, State->InstrInfo->getNumOpcodes());
  unsigned OperandCount = State->InstrInfo->get(Opcode).getNumOperands();
  for (unsigned Count : {0u, OperandCount}) {
    ContextEnvironment Context(*State);
    DecodedInst Instruction;
    Instruction.Inst.setOpcode(Opcode);
    Instruction.CanonOp = CanonicalOp::BUFFER_LOAD_B32;
    for (unsigned I = 0; I != Count; ++I)
      Instruction.Inst.addOperand(MCOperand::createImm(0));
    Error Result = handleMUBUF(*Context.Ctx, Instruction);
    ASSERT_TRUE(static_cast<bool>(Result));
    StringRef Expected =
        Count == 0 ? "buffer operand count does not match its encoding"
                   : "buffer operand must be a register";
    EXPECT_NE(toString(std::move(Result)).find(Expected.str()),
              std::string::npos);
  }

  int OffsetIndex =
      COMGR::transpiler::getNamedOperandIdx(Opcode, AMDGPU::OpName::offset);
  ASSERT_GE(OffsetIndex, 0);
  ASSERT_LT(static_cast<unsigned>(OffsetIndex), OperandCount);
  for (MCOperand Offset :
       {MCOperand::createImm(-1), MCOperand::createImm(0x800000),
        MCOperand::createImm(0xffffff), MCOperand::createImm(0x1000000),
        MCOperand::createReg(MCRegister())}) {
    ContextEnvironment Context(*State);
    DecodedInst Instruction;
    Instruction.Inst.setOpcode(Opcode);
    Instruction.CanonOp = CanonicalOp::BUFFER_LOAD_B32;
    for (unsigned I = 0; I != OperandCount; ++I)
      Instruction.Inst.addOperand(MCOperand::createImm(0));
    Instruction.Inst.getOperand(OffsetIndex) = Offset;
    Error Result = handleMUBUF(*Context.Ctx, Instruction);
    ASSERT_TRUE(static_cast<bool>(Result));
    handleAllErrors(std::move(Result), [](const RaiseFailure &Failure) {
      EXPECT_EQ(Failure.reason(),
                RaiseFailureReason::UnsupportedInstructionForm);
      EXPECT_EQ(Failure.detail(),
                "buffer offset must be an unsigned 23-bit immediate");
    });
  }
}

TEST_F(RaiseContextTest, RequiredBitsRejectUnknownAndNonzeroValues) {
  DecodedInst Instruction;
  unsigned Opcode = Mc.InstrInfo->getNumOpcodes();
  for (unsigned I = 0; I != Mc.InstrInfo->getNumOpcodes(); ++I) {
    if (Mc.InstrInfo->getName(I) == "S_ENDPGM_vi") {
      Opcode = I;
      break;
    }
  }
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  Instruction.Inst.setOpcode(Opcode);
  Instruction.Inst.addOperand(MCOperand::createImm(0));
  for (bool Unknown : {false, true}) {
    ContextEnvironment Context(Mc);
    Value *Word = Context.B.getInt32(64);
    if (Unknown)
      Word = Context.Kernel->getArg(0);
    Context.Ctx->requireZeroBits(Word, 64, Instruction, "nonzero bits");
    Error Result = Context.Ctx->validateRequiredBits();
    ASSERT_TRUE(static_cast<bool>(Result));
    EXPECT_NE(toString(std::move(Result)).find("nonzero bits"),
              std::string::npos);
  }
}

TEST_F(RaiseContextTest, SetVgprMsbUsesLowImmediateByte) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  ContextEnvironment Gfx1250(*State);

  unsigned Opcode = State->InstrInfo->getNumOpcodes();
  for (unsigned I = 0; I != State->InstrInfo->getNumOpcodes(); ++I) {
    if (State->InstrInfo->getName(I) == "S_SET_VGPR_MSB") {
      Opcode = I;
      break;
    }
  }
  ASSERT_NE(Opcode, State->InstrInfo->getNumOpcodes());

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.Inst.addOperand(MCOperand::createImm(0xABD5));
  Di.CanonOp = CanonicalOp::S_SET_VGPR_MSB;
  Di.TargetSpecificFlags = State->InstrInfo->get(Opcode).TSFlags;
  OperandResolver Resolver{*Gfx1250.Ctx, Di};

  if (Error Err = handleSOPP(*Gfx1250.Ctx, Di, Resolver))
    FAIL() << toString(std::move(Err));
  EXPECT_EQ(Gfx1250.Ctx->registers().vgprMsBs(), 0xD5);

  unsigned MoveOpcode = State->InstrInfo->getNumOpcodes();
  MCRegister Vgpr0;
  MCRegister Vgpr1;
  for (unsigned I = 0; I != State->InstrInfo->getNumOpcodes(); ++I)
    if (State->InstrInfo->getName(I) == "V_MOV_B32_e32")
      MoveOpcode = I;
  for (unsigned I = 1; I != State->RegInfo->getNumRegs(); ++I) {
    StringRef Name = State->RegInfo->getName(I);
    if (Name == "VGPR0")
      Vgpr0 = MCRegister(I);
    else if (Name == "VGPR1")
      Vgpr1 = MCRegister(I);
  }
  ASSERT_NE(MoveOpcode, State->InstrInfo->getNumOpcodes());
  ASSERT_TRUE(Vgpr0);
  ASSERT_TRUE(Vgpr1);

  DecodedInst Move;
  Move.Inst.setOpcode(MoveOpcode);
  Move.Inst.addOperand(MCOperand::createReg(Vgpr1));
  Move.Inst.addOperand(MCOperand::createReg(Vgpr0));
  Gfx1250.Ctx->registers().computeVGPRAdjust(Move);
  Expected<ParsedReg> Destination = Gfx1250.Ctx->registers().parseReg(Move, 0);
  Expected<ParsedReg> Source = Gfx1250.Ctx->registers().parseReg(Move, 1);
  ASSERT_TRUE(static_cast<bool>(Destination))
      << toString(Destination.takeError());
  ASSERT_TRUE(static_cast<bool>(Source)) << toString(Source.takeError());
  EXPECT_EQ(Destination->BaseIdx, 769u);
  EXPECT_EQ(Source->BaseIdx, 256u);
}

} // namespace
