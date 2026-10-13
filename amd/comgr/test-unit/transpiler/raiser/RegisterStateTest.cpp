//===- RegisterStateTest.cpp - register state unit tests ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/register-state.h"

#include "transpiler/common/kernel-meta.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/raiser/raise_failure.h"
#include "transpiler/raiser/wave-projection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <optional>

using namespace llvm;
using namespace COMGR::transpiler;

namespace {

unsigned findOpcode(const MCInstrInfo &MII, StringRef Name) {
  for (unsigned Opc = 0; Opc != MII.getNumOpcodes(); ++Opc)
    if (MII.getName(Opc) == Name)
      return Opc;
  return MII.getNumOpcodes();
}

MCRegister findRegister(const MCRegisterInfo &MRI, StringRef Name) {
  for (unsigned Reg = 1; Reg != MRI.getNumRegs(); ++Reg)
    if (Name == MRI.getName(Reg))
      return MCRegister(Reg);
  return MCRegister();
}

class FullWaveInvariantProjection : public ReplicationProjection {
public:
  FullWaveInvariantProjection(const MCSubtargetInfo &STI, Type *I32Ty,
                              Type *I64Ty, bool FullWaveInvariant)
      : ReplicationProjection(STI, STI, I32Ty, I64Ty) {
    ProvidesSourceWaveExecInvariant = FullWaveInvariant;
  }
};

class RegisterStateTest : public ::testing::Test {
protected:
  void SetUp() override {
    Expected<MCState> State = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
    Mc = std::move(*State);
    Env = std::make_unique<RegisterEnvironment>(Mc);
  }

  struct RegisterEnvironment {
    LLVMContext LLVMCtx;
    Module Mod;
    IRBuilder<> B;
    std::unique_ptr<WaveProjection> Projection;
    Function *Kernel;
    std::optional<RegisterState> Regs;

    explicit RegisterEnvironment(const MCState &Mc,
                                 bool FullWaveInvariant = false)
        : Mod("register_state_test", LLVMCtx), B(LLVMCtx),
          Projection(std::make_unique<FullWaveInvariantProjection>(
              *Mc.SubtargetInfo, B.getInt32Ty(), B.getInt64Ty(),
              FullWaveInvariant)),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      initialize(Mc);
    }

    RegisterEnvironment(const MCState &SourceMc, const MCState &TargetMc)
        : Mod("register_state_test", LLVMCtx), B(LLVMCtx),
          Projection(std::make_unique<WaveNativeProjection>(
              *SourceMc.SubtargetInfo, *TargetMc.SubtargetInfo, B.getInt32Ty(),
              B.getInt64Ty())),
          Kernel(Function::Create(
              FunctionType::get(B.getVoidTy(), /*isVarArg=*/false),
              Function::ExternalLinkage, "kernel", Mod)) {
      initialize(SourceMc);
    }

    void initialize(const MCState &Mc) {
      BasicBlock *Entry = BasicBlock::Create(LLVMCtx, "entry", Kernel);
      B.SetInsertPoint(Entry);
      Regs.emplace(
          cantFail(RegisterState::create(B, *Projection, Mc, KernelMeta())));
    }
  };

  static void promoteAndFold(RegisterEnvironment &Environment) {
    SmallVector<AllocaInst *> Allocas;
    Environment.Regs->collectAllocas(Allocas);
    DominatorTree DT(*Environment.Kernel);
    PromoteMemToReg(Allocas, DT);

    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (BasicBlock &BB : *Environment.Kernel) {
        for (Instruction &Instruction : make_early_inc_range(BB)) {
          if (Instruction.isTerminator()) {
            continue;
          }
          if (Constant *Folded = ConstantFoldInstruction(
                  &Instruction, Environment.Mod.getDataLayout())) {
            Instruction.replaceAllUsesWith(Folded);
            Instruction.eraseFromParent();
            Changed = true;
          }
        }
      }
    }
  }

  MCState Mc;
  std::unique_ptr<RegisterEnvironment> Env;
};

TEST_F(RegisterStateTest, ParsesArchitecturalRegisterHalves) {
  auto Check = [&](StringRef OpcodeName, StringRef Name, ParsedReg::Kind Kind,
                   unsigned Base, unsigned Width) {
    MCRegister Reg = findRegister(*Mc.RegInfo, Name);
    ASSERT_TRUE(Reg) << Name.str();
    DecodedInst Di;
    Di.Inst.setOpcode(findOpcode(*Mc.InstrInfo, OpcodeName));
    Di.Inst.addOperand(MCOperand::createReg(Reg));
    Expected<ParsedReg> Parsed = Env->Regs->parseReg(Di, 0);
    ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
    ParsedReg Pr = *Parsed;
    EXPECT_EQ(Pr.RegKind, Kind) << Name.str();
    ASSERT_TRUE(Pr.BaseIdx) << Name.str();
    EXPECT_EQ(*Pr.BaseIdx, Base) << Name.str();
    EXPECT_EQ(Pr.WidthInDwords, Width) << Name.str();
  };

  Check("S_MOV_B32", "VCC_LO", ParsedReg::VCC, 0, 1);
  Check("S_MOV_B32", "VCC_HI", ParsedReg::VCC, 1, 1);
  Check("S_MOV_B64", "VCC", ParsedReg::VCC, 0, 2);
  Check("S_MOV_B32", "FLAT_SCR_LO", ParsedReg::FLAT_SCR, 0, 1);
  Check("S_MOV_B32", "FLAT_SCR_HI", ParsedReg::FLAT_SCR, 1, 1);
  Check("S_MOV_B64", "FLAT_SCR", ParsedReg::FLAT_SCR, 0, 2);
}

TEST_F(RegisterStateTest, ParsesWideRegisterOperands) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_LOAD_DWORDX8_IMM");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  const MCInstrDesc &Descriptor = Mc.InstrInfo->get(Opcode);
  ASSERT_GT(Descriptor.getNumOperands(), 0u);
  const MCOperandInfo &OperandInfo = Descriptor.operands()[0];
  const int16_t RegisterClassID = Mc.InstrInfo->getOpRegClassID(
      OperandInfo,
      Mc.SubtargetInfo->getHwMode(MCSubtargetInfo::HwMode_RegInfo));
  ASSERT_GE(RegisterClassID, 0);
  const MCRegisterClass &RegisterClass =
      Mc.RegInfo->getRegClass(RegisterClassID);
  ASSERT_GT(RegisterClass.getNumRegs(), 0u);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.Inst.addOperand(MCOperand::createReg(RegisterClass.getRegister(0)));
  Expected<ParsedReg> Parsed = Env->Regs->parseReg(Di, 0);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
  EXPECT_EQ(Parsed->RegKind, ParsedReg::SGPR);
  EXPECT_EQ(Parsed->WidthInDwords, 8u);
}

TEST_F(RegisterStateTest, RejectsRegistersOutsideOperandClass) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_LOAD_DWORDX8_IMM_vi");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());
  const MCRegister Register = findRegister(*Mc.RegInfo, "VGPR0");
  ASSERT_TRUE(Register);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.TargetSpecificFlags = Mc.InstrInfo->get(Opcode).TSFlags;
  Di.Inst.addOperand(MCOperand::createReg(Register));
  Expected<ParsedReg> Parsed = Env->Regs->parseReg(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Parsed));
  std::string Format;
  std::string Detail;
  handleAllErrors(Parsed.takeError(), [&](const RaiseFailure &Failure) {
    ASSERT_TRUE(Failure.format());
    Format = Failure.format()->str();
    Detail = Failure.detail().str();
  });
  EXPECT_EQ(Format, "SMEM");
  EXPECT_NE(Detail.find("register-decode"), std::string::npos);
  EXPECT_NE(Detail.find("not in operand register class"), std::string::npos);
}

TEST_F(RegisterStateTest, KeepsWave32VccHighAsScratch) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  RegisterEnvironment Gfx1250(*State);
  MCRegister Reg = findRegister(*State->RegInfo, "VCC_HI");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(findOpcode(*State->InstrInfo, "S_MOV_B32"));
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<ParsedReg> Parsed = Gfx1250.Regs->parseReg(Di, 0);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
  ParsedReg Pr = *Parsed;
  EXPECT_EQ(Pr.RegKind, ParsedReg::VCC_HI_SCRATCH);
  EXPECT_EQ(Pr.WidthInDwords, 1u);
}

TEST_F(RegisterStateTest, AppliesVgprMsbsToBothVopdComponents) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  RegisterEnvironment Gfx1250(*State);

  unsigned Opc =
      findOpcode(*State->InstrInfo, "V_DUAL_ADD_F32_e32_X_ADD_F32_e32_gfx1250");
  ASSERT_NE(Opc, State->InstrInfo->getNumOpcodes());
  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Gfx1250.Regs->setVgprMsBs(0xD5);
  Gfx1250.Regs->computeVGPRAdjust(Di);

  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[0], 768u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[1], 768u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[2], 256u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[3], 256u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[4], 256u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust()[5], 256u);
  EXPECT_EQ(Gfx1250.Regs->currentVgprAdjust().size(),
            State->InstrInfo->get(Opc).getNumOperands());
}

TEST_F(RegisterStateTest, SizesVgprAdjustmentsFromDescriptor) {
  unsigned Opcode = 0;
  unsigned MaxOperands = 0;
  for (unsigned I = 0; I != Mc.InstrInfo->getNumOpcodes(); ++I) {
    unsigned NumOperands = Mc.InstrInfo->get(I).getNumOperands();
    if (NumOperands > MaxOperands) {
      Opcode = I;
      MaxOperands = NumOperands;
    }
  }
  ASSERT_GT(MaxOperands, 16u);

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Env->Regs->computeVGPRAdjust(Di);
  EXPECT_EQ(Env->Regs->currentVgprAdjust().size(), MaxOperands);
}

TEST_F(RegisterStateTest, ReportsUnsupportedRegisterOperands) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "SRC_SHARED_BASE_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<Value *> Result = Env->Regs->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Message = toString(Result.takeError());
  EXPECT_NE(Message.find("register-decode"), std::string::npos);
  EXPECT_NE(Message.find("SRC_SHARED_BASE_LO"), std::string::npos);
}

TEST_F(RegisterStateTest, ReportsEncodingFormatForOperandFailures) {
  const unsigned Opcode = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opcode, Mc.InstrInfo->getNumOpcodes());

  DecodedInst Di;
  Di.Inst.setOpcode(Opcode);
  Di.TargetSpecificFlags = Mc.InstrInfo->get(Opcode).TSFlags;
  Di.Inst.addOperand(MCOperand::createDFPImm(0));
  Expected<Value *> Result = Env->Regs->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Format;
  std::string Detail;
  handleAllErrors(Result.takeError(), [&](const RaiseFailure &Failure) {
    ASSERT_TRUE(Failure.format());
    Format = Failure.format()->str();
    Detail = Failure.detail().str();
  });
  EXPECT_EQ(Format, "SOP1");
  EXPECT_NE(Detail.find("operand-read"), std::string::npos);
}

TEST_F(RegisterStateTest, RejectsXnackMaskOperands) {
  unsigned Opc = findOpcode(*Mc.InstrInfo, "S_MOV_B32_vi");
  ASSERT_NE(Opc, Mc.InstrInfo->getNumOpcodes());
  MCRegister Reg = findRegister(*Mc.RegInfo, "XNACK_MASK_LO");
  ASSERT_TRUE(Reg);

  DecodedInst Di;
  Di.Inst.setOpcode(Opc);
  Di.Inst.addOperand(MCOperand::createReg(Reg));
  Expected<Value *> Result = Env->Regs->readOp32(Di, 0);
  ASSERT_FALSE(static_cast<bool>(Result));
  std::string Message = toString(Result.takeError());
  EXPECT_NE(Message.find("unsupported-instruction-form"), std::string::npos);
  EXPECT_NE(Message.find("XNACK_MASK_LO"), std::string::npos);
}

TEST_F(RegisterStateTest, DiscardsNullRegisterWrites) {
  Expected<MCState> State = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(State)) << toString(State.takeError());
  RegisterEnvironment Gfx1250(*State);
  BasicBlock *Block = Gfx1250.B.GetInsertBlock();
  size_t InstructionCount = Block->size();
  for (StringRef Name : {"SGPR_NULL", "SGPR_NULL_HI"}) {
    MCRegister Reg = findRegister(*State->RegInfo, Name);
    ASSERT_TRUE(Reg) << Name.str();

    DecodedInst Di;
    Di.Inst.setOpcode(findOpcode(*State->InstrInfo, "S_MOV_B32"));
    Di.Inst.addOperand(MCOperand::createReg(Reg));
    Expected<ParsedReg> Parsed = Gfx1250.Regs->parseReg(Di, 0);
    ASSERT_TRUE(static_cast<bool>(Parsed)) << toString(Parsed.takeError());
    ASSERT_EQ(Parsed->RegKind, ParsedReg::NOREG);

    Gfx1250.Regs->writeReg32(*Parsed, Gfx1250.B.getInt32(1));
    Gfx1250.Regs->writeReg64(*Parsed, Gfx1250.B.getInt64(1));
    Gfx1250.Regs->writeRegVec(
        *Parsed, ConstantVector::getSplat(ElementCount::getFixed(2),
                                          Gfx1250.B.getInt32(1)));
    Gfx1250.Regs->writeRegExecWidth(*Parsed, Gfx1250.B.getInt32(1));
  }
  EXPECT_EQ(Block->size(), InstructionCount);
}

TEST_F(RegisterStateTest, InvalidatesOverlappingPairShadows) {
  Env->Regs->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Regs->recordSgprWaveMaskI1(6, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Regs->recordSourceImageSgprPairAddr(4, 0x1000);

  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(4, false), nullptr);
  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(6, true), nullptr);

  Env->Regs->invalidateSgprWaveMaskI1(5);

  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(4, true), nullptr);
  EXPECT_NE(Env->Regs->lookupSgprWaveMaskI1(6, false), nullptr);
  EXPECT_FALSE(Env->Regs->lookupSourceImageSgprPairAddr(4));
}

TEST_F(RegisterStateTest, DropsBlockScopedFactsOnBlockEntry) {
  Env->Regs->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Regs->recordSourceImageSgprPairAddr(4, 0x1000);
  Env->Regs->setVgprMsBs(0xD5);
  ParsedReg M0;
  M0.RegKind = ParsedReg::M0;
  Env->Regs->writeReg32(M0, Env->B.getInt32(7));
  Value *OldLaneActive = Env->Regs->emitLaneActiveBit();

  BasicBlock *Successor = BasicBlock::Create(
      Env->LLVMCtx, "successor", Env->B.GetInsertBlock()->getParent());
  Env->B.CreateBr(Successor);
  Env->B.SetInsertPoint(Successor);

  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(4, true), nullptr);
  EXPECT_FALSE(Env->Regs->lookupSourceImageSgprPairAddr(4));
  EXPECT_EQ(Env->Regs->vgprMsBs(), 0);
  EXPECT_FALSE(Env->Regs->getM0Const());
  EXPECT_NE(Env->Regs->emitLaneActiveBit(), OldLaneActive);
}

TEST_F(RegisterStateTest, ProjectsMaskReadsToCurrentSourceWave) {
  Expected<MCState> SourceState = initMCState("gfx1250");
  ASSERT_TRUE(static_cast<bool>(SourceState))
      << toString(SourceState.takeError());
  RegisterEnvironment Widening(*SourceState, Mc);
  Value *Lane = Widening.Projection->emitLaneIdx(Widening.B);
  Value *UpperWave =
      Widening.B.CreateICmpUGE(Lane, Widening.B.getInt32(32), "upper_wave");
  Value *ExecMask = Widening.B.CreateSelect(UpperWave, Widening.B.getInt32(-1),
                                            Widening.B.getInt32(0));
  Widening.Regs->regFile().storeVCC(Widening.B, UpperWave);
  Widening.Regs->regFile().storeExec(Widening.B, ExecMask);

  const unsigned Opcode = findOpcode(*SourceState->InstrInfo, "S_MOV_B32");
  ASSERT_NE(Opcode, SourceState->InstrInfo->getNumOpcodes());
  SmallVector<StoreInst *> Observed;
  for (StringRef RegisterName :
       {"VCC_LO", "EXEC_LO", "SRC_VCCZ", "SRC_EXECZ"}) {
    MCRegister Register = findRegister(*SourceState->RegInfo, RegisterName);
    ASSERT_TRUE(Register) << RegisterName.str();
    DecodedInst Instruction;
    Instruction.Inst.setOpcode(Opcode);
    Instruction.TargetSpecificFlags =
        SourceState->InstrInfo->get(Opcode).TSFlags;
    Instruction.Inst.addOperand(MCOperand::createReg(Register));
    Expected<Value *> Result = Widening.Regs->readOp32(Instruction, 0);
    ASSERT_TRUE(static_cast<bool>(Result)) << toString(Result.takeError());
    AllocaInst *Output =
        Widening.B.CreateAlloca(Widening.B.getInt32Ty(), nullptr, RegisterName);
    Observed.push_back(Widening.B.CreateStore(*Result, Output));
  }
  Widening.B.CreateRetVoid();
  promoteAndFold(Widening);

  using namespace PatternMatch;
  const auto SourceWaveMask = m_Trunc(m_LShr(
      m_Intrinsic<Intrinsic::amdgcn_ballot>(m_Specific(UpperWave)),
      m_ZExt(m_And(m_Specific(Lane), m_Specific(Widening.B.getInt32(-32))))));
  EXPECT_TRUE(match(Observed[0]->getValueOperand(), SourceWaveMask));
  EXPECT_EQ(Observed[1]->getValueOperand(), ExecMask);
  EXPECT_TRUE(match(
      Observed[2]->getValueOperand(),
      m_ZExt(m_SpecificICmp(ICmpInst::ICMP_EQ, SourceWaveMask, m_Zero()))));
  EXPECT_TRUE(match(Observed[3]->getValueOperand(),
                    m_ZExt(m_SpecificICmp(ICmpInst::ICMP_EQ,
                                          m_Specific(ExecMask), m_Zero()))));
}

TEST_F(RegisterStateTest, RetainsPairWidthAcrossBlocks) {
  Env->Regs->recordSgprWaveMaskI1(2, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Regs->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  BasicBlock *Successor = BasicBlock::Create(
      Env->LLVMCtx, "successor", Env->B.GetInsertBlock()->getParent());
  Env->B.CreateBr(Successor);
  Env->B.SetInsertPoint(Successor);

  Env->Regs->invalidateSgprWaveMaskI1(3);
  Env->Regs->invalidateSgprWaveMaskI1(5);
  Value *SingleValid = Env->Regs->loadSgprWaveMaskValid(2);
  Value *PairValid = Env->Regs->loadSgprWaveMaskValid(4);
  AllocaInst *ObservedSingle =
      Env->B.CreateAlloca(Env->B.getInt1Ty(), nullptr, "observed_single");
  AllocaInst *ObservedPair =
      Env->B.CreateAlloca(Env->B.getInt1Ty(), nullptr, "observed_pair");
  StoreInst *StoreSingle = Env->B.CreateStore(SingleValid, ObservedSingle);
  StoreInst *StorePair = Env->B.CreateStore(PairValid, ObservedPair);
  Env->B.CreateRetVoid();

  promoteAndFold(*Env);
  auto *StoredSingle = dyn_cast<ConstantInt>(StoreSingle->getValueOperand());
  auto *StoredPair = dyn_cast<ConstantInt>(StorePair->getValueOperand());
  ASSERT_NE(StoredSingle, nullptr);
  ASSERT_NE(StoredPair, nullptr);
  EXPECT_TRUE(StoredSingle->isOne());
  EXPECT_TRUE(StoredPair->isZero());
}

TEST_F(RegisterStateTest, MaintainsStateOnRegisterWrites) {
  ParsedReg Sgpr;
  Sgpr.RegKind = ParsedReg::SGPR;
  Sgpr.BaseIdx = 5;
  Env->Regs->recordSgprWaveMaskI1(4, ConstantInt::getTrue(Env->LLVMCtx), true);
  Env->Regs->recordSourceImageSgprPairAddr(4, 0x1000);
  Env->Regs->writeReg32(Sgpr, Env->B.getInt32(1));
  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(4, true), nullptr);
  EXPECT_FALSE(Env->Regs->lookupSourceImageSgprPairAddr(4));

  Sgpr.BaseIdx = 8;
  Sgpr.WidthInDwords = 2;
  Env->Regs->recordSgprWaveMaskI1(8, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Regs->recordSgprWaveMaskI1(9, ConstantInt::getTrue(Env->LLVMCtx), false);
  Env->Regs->writeRegExecWidth(Sgpr, Env->B.getInt64(1));
  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(8, false), nullptr);
  EXPECT_EQ(Env->Regs->lookupSgprWaveMaskI1(9, false), nullptr);

  ParsedReg M0;
  M0.RegKind = ParsedReg::M0;
  Env->Regs->writeReg32(M0, Env->B.getInt32(7));
  EXPECT_EQ(Env->Regs->getM0Const(), 7u);
  Env->Regs->writeReg32(M0, PoisonValue::get(Env->B.getInt32Ty()));
  EXPECT_FALSE(Env->Regs->getM0Const());

  Value *OldLaneActive = Env->Regs->emitLaneActiveBit();
  ParsedReg Exec;
  Exec.RegKind = ParsedReg::EXEC;
  Exec.BaseIdx = 0;
  Env->Regs->writeReg32(Exec, Env->B.getInt32(1));
  EXPECT_NE(Env->Regs->emitLaneActiveBit(), OldLaneActive);
}

TEST_F(RegisterStateTest, InactiveSourceWavePreservesInvalidPairShadow) {
  RegisterEnvironment FullWave(Mc, true);
  FullWave.Regs->regFile().storeExec(FullWave.B, FullWave.B.getInt64(0));

  FullWave.Regs->recordSourceWaveSgprPair(0, FullWave.B.getInt64(7));
  Value *Result =
      FullWave.Regs->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *Observed =
      FullWave.B.CreateAlloca(FullWave.B.getInt64Ty(), nullptr, "observed");
  StoreInst *Store = FullWave.B.CreateStore(Result, Observed);
  FullWave.B.CreateRetVoid();

  promoteAndFold(FullWave);
  auto *Stored = dyn_cast<ConstantInt>(Store->getValueOperand());
  ASSERT_NE(Stored, nullptr);
  EXPECT_EQ(Stored->getZExtValue(), 42u);
}

TEST_F(RegisterStateTest, OwnsPerSgprShadowStorage) {
  SmallVector<AllocaInst *> RegisterFileAllocas;
  Env->Regs->regFile().collectAllocas(RegisterFileAllocas);
  SmallVector<AllocaInst *> Allocas;
  Env->Regs->collectAllocas(Allocas);
  EXPECT_EQ(Allocas.size(), RegisterFileAllocas.size() +
                                Env->Regs->regFile().Sgpr.size() * 6 + 1);
}

TEST_F(RegisterStateTest, InvalidatesPerSgprShadowStorage) {
  RegisterEnvironment FullWave(Mc, true);
  FullWave.Regs->regFile().storeExec(FullWave.B,
                                     FullWave.B.getInt64(UINT64_MAX));
  FullWave.Regs->recordSourceWaveSgprPair(0, FullWave.B.getInt64(7));
  Value *Before =
      FullWave.Regs->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *ObservedBefore = FullWave.B.CreateAlloca(
      FullWave.B.getInt64Ty(), nullptr, "observed_before");
  StoreInst *StoreBefore = FullWave.B.CreateStore(Before, ObservedBefore);

  FullWave.Regs->invalidateSgprShadows();
  Value *After =
      FullWave.Regs->materializeSourceWaveSgprPair(0, FullWave.B.getInt64(42));
  AllocaInst *ObservedAfter = FullWave.B.CreateAlloca(
      FullWave.B.getInt64Ty(), nullptr, "observed_after");
  StoreInst *StoreAfter = FullWave.B.CreateStore(After, ObservedAfter);
  FullWave.B.CreateRetVoid();

  promoteAndFold(FullWave);
  auto *StoredBefore = dyn_cast<ConstantInt>(StoreBefore->getValueOperand());
  auto *StoredAfter = dyn_cast<ConstantInt>(StoreAfter->getValueOperand());
  ASSERT_NE(StoredBefore, nullptr);
  ASSERT_NE(StoredAfter, nullptr);
  EXPECT_EQ(StoredBefore->getZExtValue(), 7u);
  EXPECT_EQ(StoredAfter->getZExtValue(), 42u);
}

} // namespace
