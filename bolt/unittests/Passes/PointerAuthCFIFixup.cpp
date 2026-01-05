//===- bolt/unittest/Passes/PointerAuthCFIFixup.cpp ----------------------===//
//
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
#include "bolt/Passes/PointerAuthCFIFixup.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

namespace opts {
extern cl::opt<bool> PrintPAuthCFIAnalyzer;
} // namespace opts

namespace {
struct PassTester : public testing::TestWithParam<Triple::ArchType> {
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

#define PREPARE_FUNC(name)                                                     \
  constexpr uint64_t FunctionAddress = 0x1000;                                 \
  BinaryFunction *BF = BC->createBinaryFunction(                               \
      name, *TextSection, FunctionAddress, /*Size=*/0, /*SymbolSize=*/0,       \
      /*Alignment=*/16);                                                       \
  /* Make sure the pass runs on the BF.*/                                      \
  BF->updateState(BinaryFunction::State::CFG);                                 \
  BF->setContainedNegateRAState();                                             \
  /* All tests need at least one BB. */                                        \
  BinaryBasicBlock *BB = BF->addBasicBlock();                                  \
  BF->addEntryPoint(*BB);                                                      \
  BB->setCFIState(0);

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

    PassManager = std::make_unique<BinaryFunctionPassManager>(*BC);
    PassManager->registerPass(
        std::make_unique<PointerAuthCFIFixup>(opts::PrintPAuthCFIAnalyzer));

    TextSection = &BC->registerOrUpdateSection(
        ".text", ELF::SHT_PROGBITS, ELF::SHF_ALLOC | ELF::SHF_EXECINSTR,
        /*Data=*/nullptr, /*Size=*/0,
        /*Alignment=*/16);
  }

  std::vector<int> findCFIOffsets(BinaryFunction &BF) {
    std::vector<int> Locations;
    int Idx = 0;
    int InstSize = 4; // AArch64
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &Inst : BB) {
        if (BC->MIB->isCFI(Inst)) {
          const MCCFIInstruction *CFI = BF.getCFIFor(Inst);
          if (CFI->getOperation() == MCCFIInstruction::OpNegateRAState)
            Locations.push_back(Idx * InstSize);
        }
        Idx++;
      }
    }
    return Locations;
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
  std::unique_ptr<BinaryFunctionPassManager> PassManager;
  BinarySection *TextSection;
};
} // namespace

TEST_P(PassTester, ExampleTest) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  ASSERT_NE(TextSection, nullptr);

  PREPARE_FUNC("ExampleFunction");

  MCInst UnsignedInst = MCInstBuilder(AArch64::ADDSXri)
                            .addReg(AArch64::X0)
                            .addReg(AArch64::X0)
                            .addImm(0)
                            .addImm(0);
  BC->MIB->setRAState(UnsignedInst, false);
  BB->addInstruction(UnsignedInst);

  MCInst SignedInst = MCInstBuilder(AArch64::ADDSXri)
                          .addReg(AArch64::X0)
                          .addReg(AArch64::X0)
                          .addImm(1)
                          .addImm(0);
  BC->MIB->setRAState(SignedInst, true);
  BB->addInstruction(SignedInst);

  Error E = PassManager->runPasses();
  EXPECT_FALSE(E);

  /* Expected layout of BF after the pass:

   .LBB0 (3 instructions, align : 1)
      Entry Point
      CFI State : 0
        00000000:   adds    x0, x0, #0x0
        00000004:   !CFI    $0      ; OpNegateRAState
        00000004:   adds    x0, x0, #0x1
      CFI State: 0
   */
  auto CFILoc = findCFIOffsets(*BF);
  EXPECT_EQ(CFILoc.size(), 1u);
  EXPECT_EQ(CFILoc[0], 4);
}

TEST_P(PassTester, fillUnknownStateInBBTest) {
  /* Check that a if BB starts with unknown RAState, we can fill the unknown
   states based on following instructions with known RAStates.
   *
   * .LBB0 (1 instructions, align : 1)
        Entry Point
        CFI State : 0
          00000000:   adds    x0, x0, #0x0
        CFI State: 0

     .LBB1 (4 instructions, align : 1)
        CFI State : 0
          00000004:   !CFI    $0      ; OpNegateRAState
          00000004:   adds    x0, x0, #0x1
          00000008:   adds    x0, x0, #0x2
          0000000c:   adds    x0, x0, #0x3
        CFI State: 0
   */
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  ASSERT_NE(TextSection, nullptr);

  PREPARE_FUNC("FuncWithUnknownStateInBB");
  BinaryBasicBlock *BB2 = BF->addBasicBlock();
  BB2->setCFIState(0);

  MCInst Unsigned = MCInstBuilder(AArch64::ADDSXri)
                        .addReg(AArch64::X0)
                        .addReg(AArch64::X0)
                        .addImm(0)
                        .addImm(0);
  BC->MIB->setRAState(Unsigned, false);
  BB->addInstruction(Unsigned);

  MCInst Unknown = MCInstBuilder(AArch64::ADDSXri)
                       .addReg(AArch64::X0)
                       .addReg(AArch64::X0)
                       .addImm(1)
                       .addImm(0);
  MCInst Unknown1 = MCInstBuilder(AArch64::ADDSXri)
                        .addReg(AArch64::X0)
                        .addReg(AArch64::X0)
                        .addImm(2)
                        .addImm(0);
  MCInst Signed = MCInstBuilder(AArch64::ADDSXri)
                      .addReg(AArch64::X0)
                      .addReg(AArch64::X0)
                      .addImm(3)
                      .addImm(0);
  BC->MIB->setRAState(Signed, true);
  BB2->addInstruction(Unknown);
  BB2->addInstruction(Unknown1);
  BB2->addInstruction(Signed);

  Error E = PassManager->runPasses();
  EXPECT_FALSE(E);

  auto CFILoc = findCFIOffsets(*BF);
  EXPECT_EQ(CFILoc.size(), 1u);
  EXPECT_EQ(CFILoc[0], 4);
  // Check that the pass set Unknown and Unknown1 to signed.
  // begin() is the CFI, begin() + 1 is Unknown, begin() + 2 is Unknown1.
  std::optional<bool> RAState = BC->MIB->getRAState(*(BB2->begin() + 1));
  EXPECT_TRUE(RAState.has_value());
  EXPECT_TRUE(*RAState);
  std::optional<bool> RAState1 = BC->MIB->getRAState(*(BB2->begin() + 2));
  EXPECT_TRUE(RAState1.has_value());
  EXPECT_TRUE(*RAState1);
}

TEST_P(PassTester, fillUnknownStubs) {
  /*
   * Stubs that are not part of the function's CFG should inherit the RAState of
   the BasicBlock before it.
   *
   * LBB1 is not part of the CFG: LBB0 jumps unconditionally to LBB2.
   * LBB1 would be a stub inserted in LongJmp in real code.
   * We do not add any NegateRAState CFIs, as other CFIs are not added either.
   * See issue #160989 for more details.
   *
   *  .LBB0 (1 instructions, align : 1)
       Entry Point
         00000000:   b       .LBB2
       Successors: .LBB2

     .LBB1 (1 instructions, align : 1)
         00000004:   ret

     .LBB2 (1 instructions, align : 1)
       Predecessors: .LBB0
          00000008:   ret
   */
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  ASSERT_NE(TextSection, nullptr);

  PREPARE_FUNC("FuncWithStub");
  BinaryBasicBlock *BB2 = BF->addBasicBlock();
  BB2->setCFIState(0);
  BinaryBasicBlock *BB3 = BF->addBasicBlock();
  BB3->setCFIState(0);

  BB->addSuccessor(BB3);

  // Jumping over BB2, to BB3.
  MCInst Jump;
  BC->MIB->createUncondBranch(Jump, BB3->getLabel(), BC->Ctx.get());
  BB->addInstruction(Jump);
  BC->MIB->setRAState(Jump, false);

  // BB2, in real code it would be a ShortJmp.
  // Unknown RAState.
  MCInst StubInst;
  BC->MIB->createReturn(StubInst);
  BB2->addInstruction(StubInst);

  // Can be any instruction.
  MCInst Ret;
  BC->MIB->createReturn(Ret);
  BB3->addInstruction(Ret);
  BC->MIB->setRAState(Ret, false);

  Error E = PassManager->runPasses();
  EXPECT_FALSE(E);

  // Check that we did not generate any NegateRAState CFIs.
  auto CFILoc = findCFIOffsets(*BF);
  EXPECT_EQ(CFILoc.size(), 0u);
}

TEST_P(PassTester, fillUnknownStubsEmpty) {
  /*
   * This test checks that BOLT can set the RAState of unknown BBs,
   * even if all previous BBs are empty, hence no PrevInst gets set.
   *
   * As this means that the current (empty) BB is the first with non-pseudo
   * instructions, the function's initialRAState should be used.
   */
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  ASSERT_NE(TextSection, nullptr);

  PREPARE_FUNC("FuncWithStub");
  BF->setInitialRAState(false);
  BinaryBasicBlock *BB2 = BF->addBasicBlock();
  BB2->setCFIState(0);

  // BB is empty.
  BB->addSuccessor(BB2);

  // BB2, in real code it would be a ShortJmp.
  // Unknown RAState.
  MCInst StubInst;
  BC->MIB->createReturn(StubInst);
  BB2->addInstruction(StubInst);

  Error E = PassManager->runPasses();
  EXPECT_FALSE(E);

  // Check that BOLT added an RAState to BB2.
  std::optional<bool> RAState = BC->MIB->getRAState(*(BB2->begin()));
  EXPECT_TRUE(RAState.has_value());
  // BB2 should be set to BF.initialRAState (false).
  EXPECT_FALSE(*RAState);
}

#ifdef AARCH64_AVAILABLE
INSTANTIATE_TEST_SUITE_P(AArch64, PassTester,
                         ::testing::Values(Triple::aarch64));
#endif
