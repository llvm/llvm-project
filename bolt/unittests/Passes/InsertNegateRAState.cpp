//===- bolt/unittest/Passes/InsertNegateRAState.cpp -----------------------===//
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
#include "bolt/Passes/InsertNegateRAStatePass.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

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
    PassManager->registerPass(std::make_unique<InsertNegateRAState>());

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

#ifdef AARCH64_AVAILABLE
INSTANTIATE_TEST_SUITE_P(AArch64, PassTester,
                         ::testing::Values(Triple::aarch64));
#endif
