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

  /// Return true when the expected \p SampleSize profile data are generated and
  /// contain all the \p ExpectedEventNames.
  bool checkEvents(uint64_t PID, size_t SampleSize,
                   const StringSet<> &ExpectedEventNames) {
    DataAggregator DA("<pseudo input>");
    DA.ParsingBuf = opts::ReadPerfEvents;
    DA.BC = BC.get();
    DataAggregator::MMapInfo MMap;
    DA.BinaryMMapInfo.insert(std::make_pair(PID, MMap));

    DA.parseSpeAsBasicEvents();

    for (auto &EE : ExpectedEventNames)
      if (!DA.EventNames.contains(EE.first()))
        return false;

    return SampleSize == DA.BasicSamples.size();
  }
};

} // namespace bolt
} // namespace llvm

// Check that DataAggregator can parseSpeAsBasicEvents for branch events when
// combined with other event types.

TEST_F(PerfSpeEventsTestHelper, SpeBranches) {
  // Check perf input with SPE branch events.
  // Example collection command:
  // ```
  // perf record -e 'arm_spe_0/branch_filter=1/u' -- BINARY
  // ```

  opts::ReadPerfEvents =
      "1234          instructions:              a002    a001\n"
      "1234          instructions:              b002    b001\n"
      "1234          instructions:              c002    c001\n"
      "1234          instructions:              d002    d001\n"
      "1234          instructions:              e002    e001\n";

  EXPECT_TRUE(checkEvents(1234, 10, {"branch-spe:"}));
}

TEST_F(PerfSpeEventsTestHelper, SpeBranchesAndCycles) {
  // Check perf input with SPE branch events and cycles.
  // Example collection command:
  // ```
  // perf record -e cycles:u -e 'arm_spe_0/branch_filter=1/u' -- BINARY
  // ```

  opts::ReadPerfEvents =
      "1234          instructions:              a002    a001\n"
      "1234              cycles:u:                 0    b001\n"
      "1234              cycles:u:                 0    c001\n"
      "1234          instructions:              d002    d001\n"
      "1234          instructions:              e002    e001\n";

  EXPECT_TRUE(checkEvents(1234, 8, {"branch-spe:", "cycles:u:"}));
}

TEST_F(PerfSpeEventsTestHelper, SpeAnyEventAndCycles) {
  // Check perf input with any SPE event type and cycles.
  // Example collection command:
  // ```
  // perf record -e cycles:u -e 'arm_spe_0//u' -- BINARY
  // ```

  opts::ReadPerfEvents =
      "1234              cycles:u:                0     a001\n"
      "1234              cycles:u:                0     b001\n"
      "1234          instructions:                0     c001\n"
      "1234          instructions:                0     d001\n"
      "1234          instructions:              e002    e001\n";

  EXPECT_TRUE(
      checkEvents(1234, 6, {"cycles:u:", "instruction-spe:", "branch-spe:"}));
}

TEST_F(PerfSpeEventsTestHelper, SpeNoBranchPairsRecorded) {
  // Check perf input that has no SPE branch pairs recorded.
  // Example collection command:
  // ```
  // perf record -e cycles:u -e 'arm_spe_0/load_filter=1,branch_filter=0/u' --
  // BINARY
  // ```

  testing::internal::CaptureStderr();
  opts::ReadPerfEvents =
      "1234          instructions:                 0    a001\n"
      "1234              cycles:u:                 0    b001\n"
      "1234          instructions:                 0    c001\n"
      "1234              cycles:u:                 0    d001\n"
      "1234          instructions:                 0    e001\n";

  EXPECT_TRUE(checkEvents(1234, 5, {"instruction-spe:", "cycles:u:"}));

  std::string Stderr = testing::internal::GetCapturedStderr();
  EXPECT_EQ(Stderr, "PERF2BOLT-WARNING: no SPE branches found\n");
}

#endif
