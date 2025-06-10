//===- bolt/unittests/Profile/PerfSpeEvents.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef AARCH64_AVAILABLE

#include "bolt/Core/BinaryContext.h"
#include "bolt/Profile/DataAggregator.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;
using namespace llvm::object;
using namespace llvm::ELF;

namespace opts {
extern cl::opt<std::string> ReadPerfEvents;
extern cl::opt<bool> ArmSPE;
} // namespace opts

namespace llvm {
namespace bolt {

/// Perform checks on perf SPE branch events combined with other SPE or perf
/// events.
struct PerfSpeEventsTestHelper : public testing::Test {
  void SetUp() override {
    initalizeLLVM();
    prepareElf();
    initializeBOLT();
  }

protected:
  using Trace = DataAggregator::Trace;

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
    EHdr->e_machine = llvm::ELF::EM_AARCH64;
    MemoryBufferRef Source(StringRef(ElfBuf, sizeof(ElfBuf)), "ELF");
    ObjFile = cantFail(ObjectFile::createObjectFile(Source));
  }

  void initializeBOLT() {
    Relocation::Arch = ObjFile->makeTriple().getArch();
    BC = cantFail(BinaryContext::createBinaryContext(
        ObjFile->makeTriple(), std::make_shared<orc::SymbolStringPool>(),
        ObjFile->getFileName(), nullptr, /*IsPIC*/ false,
        DWARFContext::create(*ObjFile.get()), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;

  // Parse and check SPE brstack as LBR.
  void parseAndCheckBrstackEvents(uint64_t PID, uint64_t From, uint64_t To,
                                  uint64_t Count, size_t SampleSize) {
    DataAggregator DA("<pseudo input>");
    DA.ParsingBuf = opts::ReadPerfEvents;
    DA.BC = BC.get();
    DataAggregator::MMapInfo MMap;
    DA.BinaryMMapInfo.insert(std::make_pair(PID, MMap));

    DA.parseBranchEvents();

    EXPECT_EQ(DA.BranchLBRs.size(), SampleSize);
    EXPECT_EQ(DA.BranchLBRs[Trace(From, To)].MispredCount, Count);
    EXPECT_EQ(DA.BranchLBRs[Trace(From, To)].TakenCount, Count);
  }
};

} // namespace bolt
} // namespace llvm

TEST_F(PerfSpeEventsTestHelper, SpeBranchesWithBrstack) {
  // Check perf input with SPE branch events as brstack format.
  // Example collection command:
  // ```
  // perf record -e 'arm_spe_0/branch_filter=1/u' -- BINARY
  // ```
  // How Bolt extracts the branch events:
  // ```
  // perf script -F pid,brstack --itrace=bl
  // ```

  opts::ArmSPE = true;
  opts::ReadPerfEvents = "  1234  0xa001/0xa002/PN/-/-/10/COND/-\n"
                         "  1234  0xb001/0xb002/P/-/-/4/RET/-\n"
                         "  1234  0xc001/0xc002/P/-/-/13/-/-\n"
                         "  1234  0xd001/0xd002/M/-/-/7/RET/-\n"
                         "  1234  0xe001/0xe002/P/-/-/14/RET/-\n"
                         "  1234  0xd001/0xd002/M/-/-/7/RET/-\n"
                         "  1234  0xf001/0xf002/MN/-/-/8/COND/-\n";

  parseAndCheckBrstackEvents(1234, 0xd001, 0xd002, 2, 6);
}

#endif
