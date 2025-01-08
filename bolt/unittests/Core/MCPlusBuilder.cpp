//===- bolt/unittest/Core/MCPlusBuilder.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef AARCH64_AVAILABLE
#include "AArch64Subtarget.h"
#endif // AARCH64_AVAILABLE

#ifdef X86_AVAILABLE
#include "X86Subtarget.h"
#endif // X86_AVAILABLE

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
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
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
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
        ObjFile->getFileName(), nullptr, true,
        DWARFContext::create(*ObjFile.get()), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
    BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(
        createMCPlusBuilder(GetParam(), BC->MIA.get(), BC->MII.get(),
                            BC->MRI.get(), BC->STI.get())));
  }

  void testRegAliases(Triple::ArchType Arch, uint64_t Register,
                      uint64_t *Aliases, size_t Count,
                      bool OnlySmaller = false) {
    if (GetParam() != Arch)
      GTEST_SKIP();

    const BitVector &BV = BC->MIB->getAliases(Register, OnlySmaller);
    ASSERT_EQ(BV.count(), Count);
    for (size_t I = 0; I < Count; ++I)
      ASSERT_TRUE(BV[Aliases[I]]);
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
  uint64_t AliasesX0[] = {AArch64::W0,    AArch64::W0_HI,
                          AArch64::X0,    AArch64::W0_W1,
                          AArch64::X0_X1, AArch64::X0_X1_X2_X3_X4_X5_X6_X7};
  size_t AliasesX0Count = sizeof(AliasesX0) / sizeof(*AliasesX0);
  testRegAliases(Triple::aarch64, AArch64::X0, AliasesX0, AliasesX0Count);
}

TEST_P(MCPlusBuilderTester, AliasSmallerX0) {
  uint64_t AliasesX0[] = {AArch64::W0, AArch64::W0_HI, AArch64::X0};
  size_t AliasesX0Count = sizeof(AliasesX0) / sizeof(*AliasesX0);
  testRegAliases(Triple::aarch64, AArch64::X0, AliasesX0, AliasesX0Count, true);
}

#endif // AARCH64_AVAILABLE

#ifdef X86_AVAILABLE

INSTANTIATE_TEST_SUITE_P(X86, MCPlusBuilderTester,
                         ::testing::Values(Triple::x86_64));

TEST_P(MCPlusBuilderTester, AliasAX) {
  uint64_t AliasesAX[] = {X86::RAX, X86::EAX, X86::AX, X86::AL, X86::AH};
  size_t AliasesAXCount = sizeof(AliasesAX) / sizeof(*AliasesAX);
  testRegAliases(Triple::x86_64, X86::AX, AliasesAX, AliasesAXCount);
}

TEST_P(MCPlusBuilderTester, AliasSmallerAX) {
  uint64_t AliasesAX[] = {X86::AX, X86::AL, X86::AH};
  size_t AliasesAXCount = sizeof(AliasesAX) / sizeof(*AliasesAX);
  testRegAliases(Triple::x86_64, X86::AX, AliasesAX, AliasesAXCount, true);
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
