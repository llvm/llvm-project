//===- bolt/Passes/RetpolineInsertion.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements RetpolineInsertion class, which replaces indirect
// branches (calls and jumps) with calls to retpolines to protect against branch
// target injection attacks.
// A unique retpoline is created for each register holding the address of the
// callee, if the callee address is in memory %r11 is used if available to
// hold the address of the callee before calling the retpoline, otherwise an
// address pattern specific retpoline is called where the callee address is
// loaded inside the retpoline.
// The user can determine when to assume %r11 available using r11-availability
// option, by default %r11 is assumed not available.
// Adding lfence instruction to the body of the speculate code is enabled by
// default and can be controlled by the user using retpoline-lfence option.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RetpolineInsertion.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bolt-retpoline"

using namespace llvm;
using namespace bolt;
namespace opts {

extern cl::OptionCategory BoltCategory;

llvm::cl::opt<bool> InsertRetpolines("insert-retpolines",
                                     cl::desc("run retpoline insertion pass"),
                                     cl::cat(BoltCategory));

llvm::cl::opt<bool>
RetpolineLfence("retpoline-lfence",
  cl::desc("determine if lfence instruction should exist in the retpoline"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<RetpolineInsertion::AvailabilityOptions> R11Availability(
    "r11-availability",
    cl::desc("determine the availability of r11 before indirect branches"),
    cl::init(RetpolineInsertion::AvailabilityOptions::NEVER),
    cl::values(clEnumValN(RetpolineInsertion::AvailabilityOptions::NEVER,
                          "never", "r11 not available"),
               clEnumValN(RetpolineInsertion::AvailabilityOptions::ALWAYS,
                          "always", "r11 available before calls and jumps"),
               clEnumValN(RetpolineInsertion::AvailabilityOptions::ABI, "abi",
                          "r11 available before calls but not before jumps")),
    cl::ZeroOrMore, cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

// Retpoline function structure:
// BB0: call BB2
// BB1: pause
//      lfence
//      jmp BB1
// BB2: mov %reg, (%rsp)
//      ret
// or
// BB2: push %r11
//      mov Address, %r11
//      mov %r11, 8(%rsp)
//      pop %r11
//      ret
BinaryFunction *createNewRetpoline(BinaryContext &BC,
                                   const std::string &RetpolineTag,
                                   const IndirectBranchInfo &BrInfo,
                                   bool R11Available) {
  auto &MIB = *BC.MIB;
  MCContext &Ctx = *BC.Ctx.get();
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Creating a new retpoline function["
                    << RetpolineTag << "]\n");

  BinaryFunction *NewRetpoline =
      BC.createInjectedBinaryFunction(RetpolineTag, true);
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBlocks(3);
  for (int I = 0; I < 3; I++) {
    MCSymbol *Symbol =
        Ctx.createNamedTempSymbol(Twine(RetpolineTag + "_BB" + to_string(I)));
    NewBlocks[I] = NewRetpoline->createBasicBlock(Symbol);
    NewBlocks[I].get()->setCFIState(0);
  }

  BinaryBasicBlock &BB0 = *NewBlocks[0].get();
  BinaryBasicBlock &BB1 = *NewBlocks[1].get();
  BinaryBasicBlock &BB2 = *NewBlocks[2].get();

  BB0.addSuccessor(&BB2, 0, 0);
  BB1.addSuccessor(&BB1, 0, 0);

  // Build BB0
  MCInst DirectCall;
  MIB.createDirectCall(DirectCall, BB2.getLabel(), &Ctx, /*IsTailCall*/ false);
  BB0.addInstruction(DirectCall);

  // Build BB1
  MCInst Pause;
  MIB.createPause(Pause);
  BB1.addInstruction(Pause);

  if (opts::RetpolineLfence) {
    MCInst Lfence;
    MIB.createLfence(Lfence);
    BB1.addInstruction(Lfence);
  }

  InstructionListType Seq;
  MIB.createShortJmp(Seq, BB1.getLabel(), &Ctx);
  BB1.addInstructions(Seq.begin(), Seq.end());

  // Build BB2
  if (BrInfo.isMem()) {
    if (R11Available) {
      MCInst StoreToStack;
      MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 0,
                            MIB.getX86R11(), 8);
      BB2.addInstruction(StoreToStack);
    } else {
      MCInst PushR11;
      MIB.createPushRegister(PushR11, MIB.getX86R11(), 8);
      BB2.addInstruction(PushR11);

      MCInst LoadCalleeAddrs;
      const IndirectBranchInfo::MemOpInfo &MemRef = BrInfo.Memory;
      MIB.createLoad(LoadCalleeAddrs, MemRef.BaseRegNum, MemRef.ScaleImm,
                     MemRef.IndexRegNum, MemRef.DispImm, MemRef.DispExpr,
                     MemRef.SegRegNum, MIB.getX86R11(), 8);

      BB2.addInstruction(LoadCalleeAddrs);

      MCInst StoreToStack;
      MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 8,
                            MIB.getX86R11(), 8);
      BB2.addInstruction(StoreToStack);

      MCInst PopR11;
      MIB.createPopRegister(PopR11, MIB.getX86R11(), 8);
      BB2.addInstruction(PopR11);
    }
  } else if (BrInfo.isReg()) {
    MCInst StoreToStack;
    MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 0,
                          BrInfo.BranchReg, 8);
    BB2.addInstruction(StoreToStack);
  } else {
    llvm_unreachable("not expected");
  }

  // return
  MCInst Return;
  MIB.createReturn(Return);
  BB2.addInstruction(Return);
  NewRetpoline->insertBasicBlocks(nullptr, std::move(NewBlocks),
                                  /* UpdateLayout */ true,
                                  /* UpdateCFIState */ false);

  NewRetpoline->updateState(BinaryFunction::State::CFG_Finalized);
  return NewRetpoline;
}

std::string createRetpolineFunctionTag(BinaryContext &BC,
                                       const IndirectBranchInfo &BrInfo,
                                       bool R11Available) {
  std::string Tag;
  llvm::raw_string_ostream TagOS(Tag);
  TagOS << "__retpoline_";

  if (BrInfo.isReg()) {
    BC.InstPrinter->printRegName(TagOS, BrInfo.BranchReg);
    TagOS << "_";
    TagOS.flush();
    return Tag;
  }

  // Memory Branch
  if (R11Available)
    return "__retpoline_r11";

  const IndirectBranchInfo::MemOpInfo &MemRef = BrInfo.Memory;

  TagOS << "mem_";

  if (MemRef.BaseRegNum != BC.MIB->getNoRegister())
    BC.InstPrinter->printRegName(TagOS, MemRef.BaseRegNum);

  TagOS << "+";
  if (MemRef.DispExpr)
    MemRef.DispExpr->print(TagOS, BC.AsmInfo.get());
  else
    TagOS << MemRef.DispImm;

  if (MemRef.IndexRegNum != BC.MIB->getNoRegister()) {
    TagOS << "+" << MemRef.ScaleImm << "*";
    BC.InstPrinter->printRegName(TagOS, MemRef.IndexRegNum);
  }

  if (MemRef.SegRegNum != BC.MIB->getNoRegister()) {
    TagOS << "_seg_";
    BC.InstPrinter->printRegName(TagOS, MemRef.SegRegNum);
  }

  TagOS.flush();
  return Tag;
}

BinaryFunction *RetpolineInsertion::getOrCreateRetpoline(
    BinaryContext &BC, const IndirectBranchInfo &BrInfo, bool R11Available) {
  const std::string RetpolineTag =
      createRetpolineFunctionTag(BC, BrInfo, R11Available);

  if (CreatedRetpolines.count(RetpolineTag))
    return CreatedRetpolines[RetpolineTag];

  return CreatedRetpolines[RetpolineTag] =
             createNewRetpoline(BC, RetpolineTag, BrInfo, R11Available);
}

void createBranchReplacement(BinaryContext &BC,
                             const IndirectBranchInfo &BrInfo,
                             bool R11Available,
                             InstructionListType &Replacement,
                             const MCSymbol *RetpolineSymbol) {
  auto &MIB = *BC.MIB;
  // Load the branch address in r11 if available
  if (BrInfo.isMem() && R11Available) {
    const IndirectBranchInfo::MemOpInfo &MemRef = BrInfo.Memory;
    MCInst LoadCalleeAddrs;
    MIB.createLoad(LoadCalleeAddrs, MemRef.BaseRegNum, MemRef.ScaleImm,
                   MemRef.IndexRegNum, MemRef.DispImm, MemRef.DispExpr,
                   MemRef.SegRegNum, MIB.getX86R11(), 8);
    Replacement.push_back(LoadCalleeAddrs);
  }

  // Call the retpoline
  MCInst RetpolineCall;
  MIB.createDirectCall(RetpolineCall, RetpolineSymbol, BC.Ctx.get(),
                       BrInfo.isJump() || BrInfo.isTailCall());

  Replacement.push_back(RetpolineCall);
}

IndirectBranchInfo::IndirectBranchInfo(MCInst &Inst, MCPlusBuilder &MIB) {
  IsCall = MIB.isCall(Inst);
  IsTailCall = MIB.isTailCall(Inst);

  if (MIB.isBranchOnMem(Inst)) {
    IsMem = true;
    std::optional<MCPlusBuilder::X86MemOperand> MO =
        MIB.evaluateX86MemoryOperand(Inst);
    if (!MO)
      llvm_unreachable("not expected");
    Memory = MO.value();
  } else if (MIB.isBranchOnReg(Inst)) {
    assert(MCPlus::getNumPrimeOperands(Inst) == 1 && "expect 1 operand");
    BranchReg = Inst.getOperand(0).getReg();
  } else {
    llvm_unreachable("unexpected instruction");
  }
}

Error RetpolineInsertion::runOnFunctions(BinaryContext &BC) {
  if (!opts::InsertRetpolines)
    return Error::success();

  assert(BC.isX86() &&
         "retpoline insertion not supported for target architecture");

  assert(BC.HasRelocations && "retpoline mode not supported in non-reloc");

  auto &MIB = *BC.MIB;
  uint32_t RetpolinedBranches = 0;
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    for (BinaryBasicBlock &BB : Function) {
      for (auto It = BB.begin(); It != BB.end(); ++It) {
        MCInst &Inst = *It;

        if (!MIB.isIndirectCall(Inst) && !MIB.isIndirectBranch(Inst))
          continue;

        IndirectBranchInfo BrInfo(Inst, MIB);
        bool R11Available = false;
        BinaryFunction *TargetRetpoline;
        InstructionListType Replacement;

        // Determine if r11 is available before this instruction
        if (BrInfo.isMem()) {
          if (MIB.hasAnnotation(Inst, "PLTCall"))
            R11Available = true;
          else if (opts::R11Availability == AvailabilityOptions::ALWAYS)
            R11Available = true;
          else if (opts::R11Availability == AvailabilityOptions::ABI)
            R11Available = BrInfo.isCall();
        }

        // If the instruction addressing pattern uses rsp and the retpoline
        // loads the callee address then displacement needs to be updated
        if (BrInfo.isMem() && !R11Available) {
          IndirectBranchInfo::MemOpInfo &MemRef = BrInfo.Memory;
          int Addend = (BrInfo.isJump() || BrInfo.isTailCall()) ? 8 : 16;
          if (MemRef.BaseRegNum == MIB.getStackPointer())
            MemRef.DispImm += Addend;
          if (MemRef.IndexRegNum == MIB.getStackPointer())
            MemRef.DispImm += Addend * MemRef.ScaleImm;
        }

        TargetRetpoline = getOrCreateRetpoline(BC, BrInfo, R11Available);

        createBranchReplacement(BC, BrInfo, R11Available, Replacement,
                                TargetRetpoline->getSymbol());

        It = BB.replaceInstruction(It, Replacement.begin(), Replacement.end());
        RetpolinedBranches++;
      }
    }
  }
  BC.outs() << "BOLT-INFO: The number of created retpoline functions is : "
            << CreatedRetpolines.size()
            << "\nBOLT-INFO: The number of retpolined branches is : "
            << RetpolinedBranches << "\n";
  return Error::success();
}

} // namespace bolt
} // namespace llvm
