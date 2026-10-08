//===- bolt/unittest/Passes/LivenessAnalysis.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef AARCH64_AVAILABLE
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#endif // AARCH64_AVAILABLE

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinaryFunctionCallGraph.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/RegAnalysis.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

namespace opts {
extern cl::opt<bool> AssumeABI;
} // namespace opts

namespace {
struct LivenessAnalysisTester
    : public testing::TestWithParam<Triple::ArchType> {
  void SetUp() override {
    initalizeLLVM();
    prepareElf();
    initializeBolt();
  }

protected:
  void initalizeLLVM() {
#define BOLT_TARGET(target)                                                    \
  LLVMInitialize##target##TargetInfo();                                        \
  LLVMInitialize##target##TargetMC();                                          \
  LLVMInitialize##target##AsmParser();                                         \
  LLVMInitialize##target##Disassembler();                                      \
  LLVMInitialize##target##Target();                                            \
  LLVMInitialize##target##AsmPrinter();

#include "bolt/Core/TargetConfig.def"
  }

  void prepareElf() {
    memcpy(ElfBuf, "\177ELF", 4);
    ELF64LE::Ehdr *EHdr = reinterpret_cast<typename ELF64LE::Ehdr *>(ElfBuf);
    EHdr->e_ident[llvm::ELF::EI_CLASS] = llvm::ELF::ELFCLASS64;
    EHdr->e_ident[llvm::ELF::EI_DATA] = llvm::ELF::ELFDATA2LSB;
    EHdr->e_machine = GetParam() == Triple::aarch64 ? EM_AARCH64 : EM_X86_64;
    MemoryBufferRef Source(StringRef(ElfBuf, sizeof(ElfBuf)), "ELF");
    ObjFile = cantFail(ObjectFile::createObjectFile(Source));
  }

  void initializeBolt() {
    Relocation::Arch = ObjFile->makeTriple().getArch();
    BC = cantFail(BinaryContext::createBinaryContext(
        ObjFile->makeTriple(), std::make_shared<orc::SymbolStringPool>(),
        ObjFile->getFileName(), nullptr, true, DWARFContext::create(*ObjFile),
        {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
    BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(
        createMCPlusBuilder(GetParam(), BC->MIA.get(), BC->MII.get(),
                            BC->MRI.get(), BC->STI.get())));
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};
} // namespace

#ifdef AARCH64_AVAILABLE

INSTANTIATE_TEST_SUITE_P(AArch64, LivenessAnalysisTester,
                         ::testing::Values(Triple::aarch64));

TEST_P(LivenessAnalysisTester, AArch64_scavengeRegFromState) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  opts::AssumeABI = true;
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  BinaryBasicBlock *EntryBB = BF->addBasicBlock();
  BinaryBasicBlock *FallThroughBB = BF->addBasicBlock();
  BinaryBasicBlock *TargetBB = BF->addBasicBlock();
  BF->addEntryPoint(*EntryBB);
  EntryBB->addSuccessor(FallThroughBB);
  EntryBB->addSuccessor(TargetBB);
  FallThroughBB->addSuccessor(TargetBB);
  EntryBB->setCFIState(0);
  FallThroughBB->setCFIState(0);
  TargetBB->setCFIState(0);

  // mov x8, #1
  MCInst MOVZXi =
      MCInstBuilder(AArch64::MOVZXi).addReg(AArch64::X8).addImm(1).addImm(0);
  // cbgt x0, #0, target
  MCInst CBGTXri = MCInstBuilder(AArch64::CBGTXri)
                       .addReg(AArch64::X0)
                       .addImm(0)
                       .addExpr(MCSymbolRefExpr::create(TargetBB->getLabel(),
                                                        *BC->Ctx.get()));
  // add x0, x8, #1
  MCInst ADDXri = MCInstBuilder(AArch64::ADDXri)
                      .addReg(AArch64::X0)
                      .addReg(AArch64::X8)
                      .addImm(1)
                      .addImm(0);
  // ret
  MCInst RET = MCInstBuilder(AArch64::RET).addReg(AArch64::LR);

  EntryBB->addInstruction(MOVZXi);
  EntryBB->addInstruction(CBGTXri);
  FallThroughBB->addInstruction(ADDXri);
  TargetBB->addInstruction(RET);

  BinaryFunctionCallGraph CG(buildCallGraph(*BC));
  RegAnalysis RA(*BC, &BC->getBinaryFunctions(), &CG);
  DataflowInfoManager Info(*BF, &RA, nullptr);

  auto II = EntryBB->begin();

  // Test that parameter registers are LiveIn.
  BitVector ParamRegs = BC->MIB->getRegsUsedAsParams();
  ASSERT_TRUE(ParamRegs.subsetOf(Info.getLivenessAnalysis().getLiveIn(*II)));

  BitVector LiveIn, LiveOut;
  // mov x8, #1 -> LiveIn = {x0}, LiveOut = {x0, x8}
  LiveIn = Info.getLivenessAnalysis().getLiveIn(*II);
  LiveOut = Info.getLivenessAnalysis().getLiveOut(*II);
  ASSERT_TRUE(LiveIn.test(AArch64::X0));
  ASSERT_FALSE(LiveIn.test(AArch64::X8));
  ASSERT_TRUE(LiveOut.test(AArch64::X0));
  ASSERT_TRUE(LiveOut.test(AArch64::X8));
  ASSERT_EQ(Info.getLivenessAnalysis().scavengeRegFromState(LiveOut),
            AArch64::X9);
  II++;
  // cbgt x0, #0, target -> LiveIn = {x0, x8}, LiveOut = {x0, x8}
  LiveIn = Info.getLivenessAnalysis().getLiveIn(*II);
  LiveOut = Info.getLivenessAnalysis().getLiveOut(*II);
  ASSERT_TRUE(LiveIn.test(AArch64::X0));
  ASSERT_TRUE(LiveIn.test(AArch64::X8));
  ASSERT_TRUE(LiveOut.test(AArch64::X0));
  ASSERT_TRUE(LiveOut.test(AArch64::X8));
  ASSERT_EQ(Info.getLivenessAnalysis().scavengeRegFromState(LiveOut),
            AArch64::X9);
  II = FallThroughBB->begin();
  // add x0, x8, #1 -> LiveIn = {x8}, LiveOut = {x0}
  LiveIn = Info.getLivenessAnalysis().getLiveIn(*II);
  LiveOut = Info.getLivenessAnalysis().getLiveOut(*II);
  ASSERT_FALSE(LiveIn.test(AArch64::X0));
  ASSERT_TRUE(LiveIn.test(AArch64::X8));
  ASSERT_TRUE(LiveOut.test(AArch64::X0));
  ASSERT_FALSE(LiveOut.test(AArch64::X8));
  ASSERT_EQ(Info.getLivenessAnalysis().scavengeRegFromState(LiveOut),
            AArch64::X8);
  II = TargetBB->begin();
  // ret -> LiveIn = {x0}, LiveOut = {x0}
  LiveIn = Info.getLivenessAnalysis().getLiveIn(*II);
  LiveOut = Info.getLivenessAnalysis().getLiveOut(*II);
  ASSERT_TRUE(LiveIn.test(AArch64::X0));
  ASSERT_FALSE(LiveIn.test(AArch64::X8));
  ASSERT_TRUE(LiveOut.test(AArch64::X0));
  ASSERT_FALSE(LiveOut.test(AArch64::X8));
  ASSERT_EQ(Info.getLivenessAnalysis().scavengeRegFromState(LiveOut),
            AArch64::X8);

  // Test that return registers are LiveOut.
  BitVector DefaultLiveOutRegs;
  BC->MIB->getDefaultLiveOut(DefaultLiveOutRegs);
  ASSERT_TRUE(
      DefaultLiveOutRegs.subsetOf(Info.getLivenessAnalysis().getLiveOut(*II)));
}

#endif // AARCH64_AVAILABLE
