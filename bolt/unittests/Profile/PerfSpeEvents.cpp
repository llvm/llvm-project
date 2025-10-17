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

/// Perform checks on perf SPE branch events.
struct PerfSpeEventsTestHelper : public testing::Test {
  void SetUp() override {
    initalizeLLVM();
    prepareElf();
    initializeBOLT();
  }

protected:
  using Trace = DataAggregator::Trace;
  using TakenBranchInfo = DataAggregator::TakenBranchInfo;

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
        DWARFContext::create(*ObjFile), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;

  /// Helper function to export lists to show the mismatch.
  void reportBrStackEventMismatch(
      const std::vector<std::pair<Trace, TakenBranchInfo>> &Traces,
      const std::vector<std::pair<Trace, TakenBranchInfo>> &ExpectedSamples) {
    llvm::errs() << "Traces items: \n";
    for (const auto &[Trace, BI] : Traces)
      llvm::errs() << "{" << Trace.Branch << ", " << Trace.From << ","
                   << Trace.To << ", " << BI.TakenCount << ", "
                   << BI.MispredCount << "}" << "\n";

    llvm::errs() << "Expected items: \n";
    for (const auto &[Trace, BI] : ExpectedSamples)
      llvm::errs() << "{" << Trace.Branch << ", " << Trace.From << ", "
                   << Trace.To << ", " << BI.TakenCount << ", "
                   << BI.MispredCount << "}" << "\n";
  }

  /// Parse and check SPE brstack as LBR.
  void parseAndCheckBrstackEvents(
      uint64_t PID,
      const std::vector<std::pair<Trace, TakenBranchInfo>> &ExpectedSamples) {
    DataAggregator DA("<pseudo input>");
    DA.ParsingBuf = opts::ReadPerfEvents;
    DA.BC = BC.get();
    DataAggregator::MMapInfo MMap;
    DA.BinaryMMapInfo.insert(std::make_pair(PID, MMap));

    DA.parseBranchEvents();

    EXPECT_EQ(DA.Traces.size(), ExpectedSamples.size());
    if (DA.Traces.size() != ExpectedSamples.size())
      reportBrStackEventMismatch(DA.Traces, ExpectedSamples);

    const auto TracesBegin = DA.Traces.begin();
    const auto TracesEnd = DA.Traces.end();
    for (const auto &BI : ExpectedSamples) {
      auto it = find_if(TracesBegin, TracesEnd,
                        [&BI](const auto &Tr) { return Tr.first == BI.first; });

      EXPECT_NE(it, TracesEnd);
      EXPECT_EQ(it->second.MispredCount, BI.second.MispredCount);
      EXPECT_EQ(it->second.TakenCount, BI.second.TakenCount);
    }
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
                         "  1234  0xc456/0xc789/P/-/-/13/-/-\n"
                         "  1234  0xd123/0xd456/M/-/-/7/RET/-\n"
                         "  1234  0xe001/0xe002/P/-/-/14/RET/-\n"
                         "  1234  0xd123/0xd456/M/-/-/7/RET/-\n"
                         "  1234  0xf001/0xf002/MN/-/-/8/COND/-\n"
                         "  1234  0xc456/0xc789/M/-/-/13/-/-\n";

  // ExpectedSamples contains the aggregated information about
  // a branch {{Branch From, To}, {TakenCount, MispredCount}}.
  // Consider this example trace: {{0xd123, 0xd456, Trace::BR_ONLY},
  // {2,2}}. This entry has a TakenCount = 2, as we have two samples for
  // (0xd123, 0xd456) in our input. It also has MispredsCount = 2,
  // as 'M' misprediction flag appears in both cases. BR_ONLY means
  // the trace only contains branch data.
  std::vector<std::pair<Trace, TakenBranchInfo>> ExpectedSamples = {
      {{0xa001, 0xa002, Trace::BR_ONLY}, {1, 0}},
      {{0xb001, 0xb002, Trace::BR_ONLY}, {1, 0}},
      {{0xc456, 0xc789, Trace::BR_ONLY}, {2, 1}},
      {{0xd123, 0xd456, Trace::BR_ONLY}, {2, 2}},
      {{0xe001, 0xe002, Trace::BR_ONLY}, {1, 0}},
      {{0xf001, 0xf002, Trace::BR_ONLY}, {1, 1}}};

  parseAndCheckBrstackEvents(1234, ExpectedSamples);
}

TEST_F(PerfSpeEventsTestHelper, SpeBranchesWithBrstackAndPbt) {
  // Check perf input with SPE branch events as brstack format by
  // combining with the previous branch target address (named as PBT).
  // Example collection command:
  // ```
  // perf record -e 'arm_spe_0/branch_filter=1/u' -- BINARY
  // ```
  // How Bolt extracts the branch events:
  // ```
  // perf script -F pid,brstack --itrace=bl
  // ```

  opts::ArmSPE = true;
  opts::ReadPerfEvents =
      // "<PID> <SRC>/<DEST>/PN/-/-/10/COND/- <NULL>/<PBT>/-/-/-/0//-\n"
      "  4567  0xa002/0xa003/PN/-/-/10/COND/- 0x0/0xa001/-/-/-/0//-\n"
      "  4567  0xb002/0xb003/P/-/-/4/RET/- 0x0/0xb001/-/-/-/0//-\n"
      "  4567  0xc456/0xc789/P/-/-/13/-/- 0x0/0xc123/-/-/-/0//-\n"
      "  4567  0xd456/0xd789/M/-/-/7/RET/- 0x0/0xd123/-/-/-/0//-\n"
      "  4567  0xe005/0xe009/P/-/-/14/RET/- 0x0/0xe001/-/-/-/0//-\n"
      "  4567  0xd456/0xd789/M/-/-/7/RET/- 0x0/0xd123/-/-/-/0//-\n"
      "  4567  0xf002/0xf003/MN/-/-/8/COND/- 0x0/0xf001/-/-/-/0//-\n"
      "  4567  0xc456/0xc789/P/-/-/13/-/- 0x0/0xc123/-/-/-/0//-\n";

  // ExpectedSamples contains the aggregated information about
  // a branch {{From, To, TraceTo}, {TakenCount, MispredCount}}.
  // When the SPE previous branch target address (named as PBT)
  // feature is available, an SPE sample by combining this PBT feature,
  // has two entries.
  // Arm SPE records SRC/DEST addresses of the latest sampled branch operation,
  // and it stores into the first entry. PBT records the target address of
  // most recently taken branch in program order before the sampled operation,
  // it places into the second entry.
  // They are formed a chain of two consecutive branches.
  // Where:
  //   - The previous branch operation (PBT) is always taken.
  //   - In SPE entry, the current source branch (SRC) may be either
  //     fall-through or taken.
  //   - The target address (DEST) of the recorded
  //     branch operation is always what was architecturally executed.
  // However PBT lacks associated information such as branch
  // source address, branch type, and prediction bit.
  // Considering this Trace pair:
  //  {{0xd456, 0xd789, Trace::BR_ONLY}, {2, 2}},
  //    {{0x0, 0xd123, 0xd456}, {2, 0}}
  // For SPE trace please see the description above.
  // The second entry is the PBT trace:
  // {{0x0, 0xd123, 0xd456}, {2, 0}}.
  // The PBT entry has a TakenCount = 2, as we have two samples for
  // (0x0, 0xd123) entry in our input. The 'MispredsCount = 0' is
  // always zero, because it lacks prediction information.
  // It also has no information about source branch address therefore
  // Bolt doesn't evaluate the 'From' field, and leaves it as zero (0x0).
  // TraceTo = 0xc456, means the execution jumped from
  // 0xc123 (PBT) to 0xc456 (SRC), and jumped further to 0xd789 (DEST).
  std::vector<std::pair<Trace, TakenBranchInfo>> ExpectedSamples = {
      {{0xa002, 0xa003, Trace::BR_ONLY}, {1, 0}},
      {{0x0, 0xa001, 0xa002}, {1, 0}},
      {{0xb002, 0xb003, Trace::BR_ONLY}, {1, 0}},
      {{0x0, 0xb001, 0xb002}, {1, 0}},
      {{0xc456, 0xc789, Trace::BR_ONLY}, {2, 0}},
      {{0x0, 0xc123, 0xc456}, {2, 0}},
      {{0xd456, 0xd789, Trace::BR_ONLY}, {2, 2}},
      {{0x0, 0xd123, 0xd456}, {2, 0}},
      {{0xe005, 0xe009, Trace::BR_ONLY}, {1, 0}},
      {{0x0, 0xe001, 0xe005}, {1, 0}},
      {{0xf002, 0xf003, Trace::BR_ONLY}, {1, 1}},
      {{0x0, 0xf001, 0xf002}, {1, 0}}};

  parseAndCheckBrstackEvents(4567, ExpectedSamples);
}

#endif
