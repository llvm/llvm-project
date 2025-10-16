//===- bolt/unittest/Core/MCPlusBuilder.cpp -------------------------------===//
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

#ifdef X86_AVAILABLE
#include "X86Subtarget.h"
#endif // X86_AVAILABLE

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
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

namespace {
struct MCPlusBuilderTester : public testing::TestWithParam<Triple::ArchType> {
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

  void assertRegMask(const BitVector &RegMask,
                     std::initializer_list<MCPhysReg> ExpectedRegs) {
    ASSERT_EQ(RegMask.count(), ExpectedRegs.size());
    for (MCPhysReg Reg : ExpectedRegs)
      ASSERT_TRUE(RegMask[Reg]) << "Expected " << BC->MRI->getName(Reg) << ".";
  }

  void assertRegMask(std::function<void(BitVector &)> FillRegMask,
                     std::initializer_list<MCPhysReg> ExpectedRegs) {
    BitVector RegMask(BC->MRI->getNumRegs());
    FillRegMask(RegMask);
    assertRegMask(RegMask, ExpectedRegs);
  }

  void testRegAliases(Triple::ArchType Arch, uint64_t Register,
                      std::initializer_list<MCPhysReg> ExpectedAliases,
                      bool OnlySmaller = false) {
    if (GetParam() != Arch)
      GTEST_SKIP();

    const BitVector &BV = BC->MIB->getAliases(Register, OnlySmaller);
    assertRegMask(BV, ExpectedAliases);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};
} // namespace

#ifdef AARCH64_AVAILABLE

INSTANTIATE_TEST_SUITE_P(AArch64, MCPlusBuilderTester,
                         ::testing::Values(Triple::aarch64));

TEST_P(MCPlusBuilderTester, AliasX0) {
  testRegAliases(Triple::aarch64, AArch64::X0,
                 {AArch64::W0, AArch64::W0_HI, AArch64::X0, AArch64::W0_W1,
                  AArch64::X0_X1, AArch64::X0_X1_X2_X3_X4_X5_X6_X7});
}

TEST_P(MCPlusBuilderTester, AliasSmallerX0) {
  testRegAliases(Triple::aarch64, AArch64::X0,
                 {AArch64::W0, AArch64::W0_HI, AArch64::X0},
                 /*OnlySmaller=*/true);
}

TEST_P(MCPlusBuilderTester, AArch64_CmpJE) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  std::unique_ptr<BinaryBasicBlock> BB = BF->createBasicBlock();

  InstructionListType Instrs =
      BC->MIB->createCmpJE(AArch64::X0, 2, BB->getLabel(), BC->Ctx.get());
  BB->addInstructions(Instrs.begin(), Instrs.end());
  BB->addSuccessor(BB.get());

  auto II = BB->begin();
  ASSERT_EQ(II->getOpcode(), AArch64::SUBSXri);
  ASSERT_EQ(II->getOperand(0).getReg(), AArch64::XZR);
  ASSERT_EQ(II->getOperand(1).getReg(), AArch64::X0);
  ASSERT_EQ(II->getOperand(2).getImm(), 2);
  ASSERT_EQ(II->getOperand(3).getImm(), 0);
  II++;
  ASSERT_EQ(II->getOpcode(), AArch64::Bcc);
  ASSERT_EQ(II->getOperand(0).getImm(), AArch64CC::EQ);
  const MCSymbol *Label = BC->MIB->getTargetSymbol(*II, 1);
  ASSERT_EQ(Label, BB->getLabel());
}

TEST_P(MCPlusBuilderTester, AArch64_CmpJNE) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  std::unique_ptr<BinaryBasicBlock> BB = BF->createBasicBlock();

  InstructionListType Instrs =
      BC->MIB->createCmpJNE(AArch64::X0, 2, BB->getLabel(), BC->Ctx.get());
  BB->addInstructions(Instrs.begin(), Instrs.end());
  BB->addSuccessor(BB.get());

  auto II = BB->begin();
  ASSERT_EQ(II->getOpcode(), AArch64::SUBSXri);
  ASSERT_EQ(II->getOperand(0).getReg(), AArch64::XZR);
  ASSERT_EQ(II->getOperand(1).getReg(), AArch64::X0);
  ASSERT_EQ(II->getOperand(2).getImm(), 2);
  ASSERT_EQ(II->getOperand(3).getImm(), 0);
  II++;
  ASSERT_EQ(II->getOpcode(), AArch64::Bcc);
  ASSERT_EQ(II->getOperand(0).getImm(), AArch64CC::NE);
  const MCSymbol *Label = BC->MIB->getTargetSymbol(*II, 1);
  ASSERT_EQ(Label, BB->getLabel());
}

TEST_P(MCPlusBuilderTester, testAccessedRegsImplicitDef) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // adds x0, x5, #42
  MCInst Inst = MCInstBuilder(AArch64::ADDSXri)
                    .addReg(AArch64::X0)
                    .addReg(AArch64::X5)
                    .addImm(42)
                    .addImm(0);

  assertRegMask([&](BitVector &BV) { BC->MIB->getClobberedRegs(Inst, BV); },
                {AArch64::NZCV, AArch64::W0, AArch64::X0, AArch64::W0_HI,
                 AArch64::X0_X1_X2_X3_X4_X5_X6_X7, AArch64::W0_W1,
                 AArch64::X0_X1});

  assertRegMask(
      [&](BitVector &BV) { BC->MIB->getTouchedRegs(Inst, BV); },
      {AArch64::NZCV, AArch64::W0, AArch64::W5, AArch64::X0, AArch64::X5,
       AArch64::W0_HI, AArch64::W5_HI, AArch64::X0_X1_X2_X3_X4_X5_X6_X7,
       AArch64::X2_X3_X4_X5_X6_X7_X8_X9, AArch64::X4_X5_X6_X7_X8_X9_X10_X11,
       AArch64::W0_W1, AArch64::W4_W5, AArch64::X0_X1, AArch64::X4_X5});

  assertRegMask([&](BitVector &BV) { BC->MIB->getWrittenRegs(Inst, BV); },
                {AArch64::NZCV, AArch64::W0, AArch64::X0, AArch64::W0_HI});

  assertRegMask([&](BitVector &BV) { BC->MIB->getUsedRegs(Inst, BV); },
                {AArch64::W5, AArch64::X5, AArch64::W5_HI});

  assertRegMask([&](BitVector &BV) { BC->MIB->getSrcRegs(Inst, BV); },
                {AArch64::W5, AArch64::X5, AArch64::W5_HI});
}

TEST_P(MCPlusBuilderTester, testAccessedRegsImplicitUse) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // b.eq <label>
  MCInst Inst =
      MCInstBuilder(AArch64::Bcc)
          .addImm(AArch64CC::EQ)
          .addImm(0); // <label> - should be Expr, but immediate 0 works too.

  assertRegMask([&](BitVector &BV) { BC->MIB->getClobberedRegs(Inst, BV); },
                {});

  assertRegMask([&](BitVector &BV) { BC->MIB->getTouchedRegs(Inst, BV); },
                {AArch64::NZCV});

  assertRegMask([&](BitVector &BV) { BC->MIB->getWrittenRegs(Inst, BV); }, {});

  assertRegMask([&](BitVector &BV) { BC->MIB->getUsedRegs(Inst, BV); },
                {AArch64::NZCV});

  assertRegMask([&](BitVector &BV) { BC->MIB->getSrcRegs(Inst, BV); },
                {AArch64::NZCV});
}

TEST_P(MCPlusBuilderTester, testAccessedRegsMultipleDefs) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // ldr x0, [x5], #16
  MCInst Inst = MCInstBuilder(AArch64::LDRXpost)
                    .addReg(AArch64::X5)
                    .addReg(AArch64::X0)
                    .addReg(AArch64::X5)
                    .addImm(16);

  assertRegMask(
      [&](BitVector &BV) { BC->MIB->getClobberedRegs(Inst, BV); },
      {AArch64::W0, AArch64::W5, AArch64::X0, AArch64::X5, AArch64::W0_HI,
       AArch64::W5_HI, AArch64::X0_X1_X2_X3_X4_X5_X6_X7,
       AArch64::X2_X3_X4_X5_X6_X7_X8_X9, AArch64::X4_X5_X6_X7_X8_X9_X10_X11,
       AArch64::W0_W1, AArch64::W4_W5, AArch64::X0_X1, AArch64::X4_X5});

  assertRegMask(
      [&](BitVector &BV) { BC->MIB->getTouchedRegs(Inst, BV); },
      {AArch64::W0, AArch64::W5, AArch64::X0, AArch64::X5, AArch64::W0_HI,
       AArch64::W5_HI, AArch64::X0_X1_X2_X3_X4_X5_X6_X7,
       AArch64::X2_X3_X4_X5_X6_X7_X8_X9, AArch64::X4_X5_X6_X7_X8_X9_X10_X11,
       AArch64::W0_W1, AArch64::W4_W5, AArch64::X0_X1, AArch64::X4_X5});

  assertRegMask([&](BitVector &BV) { BC->MIB->getWrittenRegs(Inst, BV); },
                {AArch64::W0, AArch64::X0, AArch64::W0_HI, AArch64::W5,
                 AArch64::X5, AArch64::W5_HI});

  assertRegMask([&](BitVector &BV) { BC->MIB->getUsedRegs(Inst, BV); },
                {AArch64::W5, AArch64::X5, AArch64::W5_HI});

  assertRegMask([&](BitVector &BV) { BC->MIB->getSrcRegs(Inst, BV); },
                {AArch64::W5, AArch64::X5, AArch64::W5_HI});
}

TEST_P(MCPlusBuilderTester, AArch64_Psign_Pauth_variants) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  MCInst Paciasp = MCInstBuilder(AArch64::PACIASP);
  MCInst Pacibsp = MCInstBuilder(AArch64::PACIBSP);
  ASSERT_TRUE(BC->MIB->isPSignOnLR(Paciasp));
  ASSERT_TRUE(BC->MIB->isPSignOnLR(Pacibsp));

  MCInst PaciaSPLR =
      MCInstBuilder(AArch64::PACIA).addReg(AArch64::LR).addReg(AArch64::SP);
  MCInst PacibSPLR =
      MCInstBuilder(AArch64::PACIB).addReg(AArch64::LR).addReg(AArch64::SP);
  ASSERT_TRUE(BC->MIB->isPSignOnLR(PaciaSPLR));
  ASSERT_TRUE(BC->MIB->isPSignOnLR(PacibSPLR));

  MCInst PacizaX5 = MCInstBuilder(AArch64::PACIZA).addReg(AArch64::X5);
  MCInst PacizbX5 = MCInstBuilder(AArch64::PACIZB).addReg(AArch64::X5);
  ASSERT_FALSE(BC->MIB->isPSignOnLR(PacizaX5));
  ASSERT_FALSE(BC->MIB->isPSignOnLR(PacizbX5));

  MCInst Paciaz = MCInstBuilder(AArch64::PACIZA).addReg(AArch64::LR);
  MCInst Pacibz = MCInstBuilder(AArch64::PACIZB).addReg(AArch64::LR);
  ASSERT_TRUE(BC->MIB->isPSignOnLR(Paciaz));
  ASSERT_TRUE(BC->MIB->isPSignOnLR(Pacibz));

  MCInst Pacia1716 = MCInstBuilder(AArch64::PACIA1716);
  MCInst Pacib1716 = MCInstBuilder(AArch64::PACIB1716);
  ASSERT_FALSE(BC->MIB->isPSignOnLR(Pacia1716));
  ASSERT_FALSE(BC->MIB->isPSignOnLR(Pacib1716));

  MCInst Pacia171615 = MCInstBuilder(AArch64::PACIA171615);
  MCInst Pacib171615 = MCInstBuilder(AArch64::PACIB171615);
  ASSERT_FALSE(BC->MIB->isPSignOnLR(Pacia171615));
  ASSERT_FALSE(BC->MIB->isPSignOnLR(Pacib171615));

  MCInst Autiasp = MCInstBuilder(AArch64::AUTIASP);
  MCInst Autibsp = MCInstBuilder(AArch64::AUTIBSP);
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(Autiasp));
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(Autibsp));

  MCInst AutiaSPLR =
      MCInstBuilder(AArch64::AUTIA).addReg(AArch64::LR).addReg(AArch64::SP);
  MCInst AutibSPLR =
      MCInstBuilder(AArch64::AUTIB).addReg(AArch64::LR).addReg(AArch64::SP);
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(AutiaSPLR));
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(AutibSPLR));

  MCInst AutizaX5 = MCInstBuilder(AArch64::AUTIZA).addReg(AArch64::X5);
  MCInst AutizbX5 = MCInstBuilder(AArch64::AUTIZB).addReg(AArch64::X5);
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(AutizaX5));
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(AutizbX5));

  MCInst Autiaz = MCInstBuilder(AArch64::AUTIZA).addReg(AArch64::LR);
  MCInst Autibz = MCInstBuilder(AArch64::AUTIZB).addReg(AArch64::LR);
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(Autiaz));
  ASSERT_TRUE(BC->MIB->isPAuthOnLR(Autibz));

  MCInst Autia1716 = MCInstBuilder(AArch64::AUTIA1716);
  MCInst Autib1716 = MCInstBuilder(AArch64::AUTIB1716);
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Autia1716));
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Autib1716));

  MCInst Autia171615 = MCInstBuilder(AArch64::AUTIA171615);
  MCInst Autib171615 = MCInstBuilder(AArch64::AUTIB171615);
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Autia171615));
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Autib171615));

  MCInst Retaa = MCInstBuilder(AArch64::RETAA);
  MCInst Retab = MCInstBuilder(AArch64::RETAB);
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Retaa));
  ASSERT_FALSE(BC->MIB->isPAuthOnLR(Retab));
  ASSERT_TRUE(BC->MIB->isPAuthAndRet(Retaa));
  ASSERT_TRUE(BC->MIB->isPAuthAndRet(Retab));
}

#endif // AARCH64_AVAILABLE

#ifdef X86_AVAILABLE

INSTANTIATE_TEST_SUITE_P(X86, MCPlusBuilderTester,
                         ::testing::Values(Triple::x86_64));

TEST_P(MCPlusBuilderTester, AliasAX) {
  testRegAliases(Triple::x86_64, X86::AX,
                 {X86::RAX, X86::EAX, X86::AX, X86::AL, X86::AH});
}

TEST_P(MCPlusBuilderTester, AliasSmallerAX) {
  testRegAliases(Triple::x86_64, X86::AX, {X86::AX, X86::AL, X86::AH},
                 /*OnlySmaller=*/true);
}

TEST_P(MCPlusBuilderTester, ReplaceRegWithImm) {
  if (GetParam() != Triple::x86_64)
    GTEST_SKIP();
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  std::unique_ptr<BinaryBasicBlock> BB = BF->createBasicBlock();
  MCInst Inst; // cmpl    %eax, %ebx
  Inst.setOpcode(X86::CMP32rr);
  Inst.addOperand(MCOperand::createReg(X86::EAX));
  Inst.addOperand(MCOperand::createReg(X86::EBX));
  auto II = BB->addInstruction(Inst);
  bool Replaced = BC->MIB->replaceRegWithImm(*II, X86::EBX, 1);
  ASSERT_TRUE(Replaced);
  ASSERT_EQ(II->getOpcode(), X86::CMP32ri8);
  ASSERT_EQ(II->getOperand(0).getReg(), X86::EAX);
  ASSERT_EQ(II->getOperand(1).getImm(), 1);
}

TEST_P(MCPlusBuilderTester, X86_CmpJE) {
  if (GetParam() != Triple::x86_64)
    GTEST_SKIP();
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  std::unique_ptr<BinaryBasicBlock> BB = BF->createBasicBlock();

  InstructionListType Instrs =
      BC->MIB->createCmpJE(X86::EAX, 2, BB->getLabel(), BC->Ctx.get());
  BB->addInstructions(Instrs.begin(), Instrs.end());
  BB->addSuccessor(BB.get());

  auto II = BB->begin();
  ASSERT_EQ(II->getOpcode(), X86::CMP64ri8);
  ASSERT_EQ(II->getOperand(0).getReg(), X86::EAX);
  ASSERT_EQ(II->getOperand(1).getImm(), 2);
  II++;
  ASSERT_EQ(II->getOpcode(), X86::JCC_1);
  const MCSymbol *Label = BC->MIB->getTargetSymbol(*II, 0);
  ASSERT_EQ(Label, BB->getLabel());
  ASSERT_EQ(II->getOperand(1).getImm(), X86::COND_E);
}

TEST_P(MCPlusBuilderTester, X86_CmpJNE) {
  if (GetParam() != Triple::x86_64)
    GTEST_SKIP();
  BinaryFunction *BF = BC->createInjectedBinaryFunction("BF", true);
  std::unique_ptr<BinaryBasicBlock> BB = BF->createBasicBlock();

  InstructionListType Instrs =
      BC->MIB->createCmpJNE(X86::EAX, 2, BB->getLabel(), BC->Ctx.get());
  BB->addInstructions(Instrs.begin(), Instrs.end());
  BB->addSuccessor(BB.get());

  auto II = BB->begin();
  ASSERT_EQ(II->getOpcode(), X86::CMP64ri8);
  ASSERT_EQ(II->getOperand(0).getReg(), X86::EAX);
  ASSERT_EQ(II->getOperand(1).getImm(), 2);
  II++;
  ASSERT_EQ(II->getOpcode(), X86::JCC_1);
  const MCSymbol *Label = BC->MIB->getTargetSymbol(*II, 0);
  ASSERT_EQ(Label, BB->getLabel());
  ASSERT_EQ(II->getOperand(1).getImm(), X86::COND_NE);
}

#endif // X86_AVAILABLE

TEST_P(MCPlusBuilderTester, Annotation) {
  MCInst Inst;
  BC->MIB->createTailCall(Inst, BC->Ctx->createNamedTempSymbol(),
                          BC->Ctx.get());
  MCSymbol *LPSymbol = BC->Ctx->createNamedTempSymbol("LP");
  uint64_t Value = INT32_MIN;
  // Test encodeAnnotationImm using this indirect way
  BC->MIB->addEHInfo(Inst, MCPlus::MCLandingPad(LPSymbol, Value));
  // Round-trip encoding-decoding check for negative values
  std::optional<MCPlus::MCLandingPad> EHInfo = BC->MIB->getEHInfo(Inst);
  ASSERT_TRUE(EHInfo.has_value());
  MCPlus::MCLandingPad LP = EHInfo.value();
  uint64_t DecodedValue = LP.second;
  ASSERT_EQ(Value, DecodedValue);

  // Large int64 should trigger an out of range assertion
  Value = 0x1FF'FFFF'FFFF'FFFFULL;
  Inst.clear();
  BC->MIB->createTailCall(Inst, BC->Ctx->createNamedTempSymbol(),
                          BC->Ctx.get());
  ASSERT_DEATH(BC->MIB->addEHInfo(Inst, MCPlus::MCLandingPad(LPSymbol, Value)),
               "annotation value out of range");
}
