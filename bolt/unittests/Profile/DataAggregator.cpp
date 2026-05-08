//===- bolt/unittests/Profile/DataAggregator.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/DataAggregator.h"
#include "bolt/Core/BinaryContext.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;

namespace opts {
extern cl::opt<bool> ReadPreAggregated;
} // namespace opts

namespace llvm {
namespace bolt {

/// Test helper with friend access to DataAggregator internals.
/// Used for both parseHexField tests (no BC needed) and pre-aggregated
/// parsing tests (BC needed, X86-only).
struct PreAggregatedTestHelper : public testing::Test {
  void SetUp() override { opts::ReadPreAggregated = true; }

protected:
  using Trace = DataAggregator::Trace;
  using TakenBranchInfo = DataAggregator::TakenBranchInfo;

  /// Parse a hex field from input string.
  ErrorOr<uint64_t> parseHex(StringRef Input) {
    DataAggregator DA("<pseudo input>");
    DA.setParsingBuffer(Input);
    return DA.parseHexField(' ', /*EndNl=*/true);
  }

  /// Parse pre-aggregated input and return collected Traces.
  /// Requires BC to be initialized (X86-only tests).
  void parseAndCollectTraces(
      StringRef Input, std::vector<std::pair<Trace, TakenBranchInfo>> &Result) {
    DataAggregator DA("<pseudo input>");
    DA.BC = BC.get();
    DA.setParsingBuffer(Input);
    std::error_code EC = DA.parsePreAggregatedLBRSamples();
    ASSERT_FALSE(EC);
    Result = std::move(DA.Traces);
  }

  /// Initialize target and BinaryContext for pre-aggregated tests.
  void initializeBOLTForX86() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();

    memcpy(ElfBuf, "\177ELF", 4);
    ELF64LE::Ehdr *EHdr = reinterpret_cast<typename ELF64LE::Ehdr *>(ElfBuf);
    EHdr->e_ident[llvm::ELF::EI_CLASS] = llvm::ELF::ELFCLASS64;
    EHdr->e_ident[llvm::ELF::EI_DATA] = llvm::ELF::ELFDATA2LSB;
    EHdr->e_machine = llvm::ELF::EM_X86_64;
    MemoryBufferRef Source(StringRef(ElfBuf, sizeof(ElfBuf)), "ELF");
    ObjFile = cantFail(ObjectFile::createObjectFile(Source));

    Relocation::Arch = ObjFile->makeTriple().getArch();
    BC = cantFail(BinaryContext::createBinaryContext(
        ObjFile->makeTriple(), std::make_shared<orc::SymbolStringPool>(),
        ObjFile->getFileName(), nullptr, /*IsPIC*/ false,
        DWARFContext::create(*ObjFile), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<object::ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};

} // namespace bolt
} // namespace llvm

TEST(DataAggregatorTest, buildID) {
  opts::ReadPreAggregated = true;

  DataAggregator DA("<pseudo input>");
  std::optional<StringRef> FileName;

  DA.setParsingBuffer("");
  ASSERT_FALSE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("1234");
  ASSERT_FALSE(FileName);

  StringRef PartialValidBuildIDs = "     File0\n"
                                   "1111 File1\n"
                                   "     File2\n";
  DA.setParsingBuffer(PartialValidBuildIDs);
  ASSERT_FALSE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("0000");
  ASSERT_FALSE(FileName);
  FileName = DA.getFileNameForBuildID("1111");
  ASSERT_EQ(*FileName, "File1");

  StringRef AllValidBuildIDs = "0000 File0\n"
                               "1111 File1\n"
                               "2222 File2\n"
                               "333  File3\n";
  DA.setParsingBuffer(AllValidBuildIDs);
  ASSERT_TRUE(DA.hasAllBuildIDs());
  FileName = DA.getFileNameForBuildID("1234");
  ASSERT_FALSE(FileName);
  FileName = DA.getFileNameForBuildID("2222");
  ASSERT_EQ(*FileName, "File2");
  FileName = DA.getFileNameForBuildID("333");
  ASSERT_EQ(*FileName, "File3");
}

TEST_F(PreAggregatedTestHelper, parseHexField) {
  auto Res = parseHex("4b196f\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, 0x4b196fULL);

  Res = parseHex("ffffffffffffffff\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, Trace::BR_ONLY);

  // -1 → UINT64_MAX (BR_ONLY / FT_ONLY).
  Res = parseHex("-1\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, Trace::BR_ONLY);

  // -2 → UINT64_MAX - 1 (FT_EXTERNAL_ORIGIN).
  Res = parseHex("-2\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, Trace::FT_EXTERNAL_ORIGIN);

  // -3 → UINT64_MAX - 2 (FT_EXTERNAL_RETURN).
  Res = parseHex("-3\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, Trace::FT_EXTERNAL_RETURN);

  Res = parseHex("0\n");
  ASSERT_TRUE(!!Res);
  EXPECT_EQ(*Res, 0ULL);
}

#ifdef X86_AVAILABLE

namespace llvm {
namespace bolt {

/// Fixture that adds X86 BinaryContext initialization on top of
/// PreAggregatedTestHelper.
struct PreAggregatedX86TestHelper : PreAggregatedTestHelper {
  void SetUp() override {
    PreAggregatedTestHelper::SetUp();
    initializeBOLTForX86();
  }
};

} // namespace bolt
} // namespace llvm

TEST_F(PreAggregatedX86TestHelper, BranchEntry) {
  // B <from> <to> <count> <mispred>
  // Trace: {from, to, BR_ONLY}
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("B 4b196f 4b19e0 2 3\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.From, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].first.To, Trace::BR_ONLY);
  EXPECT_EQ(Traces[0].second.TakenCount, 2u);
  EXPECT_EQ(Traces[0].second.MispredCount, 3u);
}

TEST_F(PreAggregatedX86TestHelper, FallThrough) {
  // F <from> <to> <count>
  // Trace: {FT_ONLY, from, to}
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("F 4b196f 4b19e0 5\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, Trace::FT_ONLY);
  EXPECT_EQ(Traces[0].first.From, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.To, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].second.TakenCount, 5u);
}

TEST_F(PreAggregatedX86TestHelper, FallThroughExternalOrigin) {
  // f <from> <to> <count>
  // Trace: {FT_EXTERNAL_ORIGIN, from, to}
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("f 4b196f 4b19e0 3\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, Trace::FT_EXTERNAL_ORIGIN);
  EXPECT_EQ(Traces[0].first.From, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.To, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].second.TakenCount, 3u);
}

TEST_F(PreAggregatedX86TestHelper, FallThroughExternalReturn) {
  // r <from> <to> <count>
  // Trace: {FT_EXTERNAL_RETURN, from, to}
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("r 4b196f 4b19e0 7\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, Trace::FT_EXTERNAL_RETURN);
  EXPECT_EQ(Traces[0].first.From, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.To, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].second.TakenCount, 7u);
}

TEST_F(PreAggregatedX86TestHelper, TraceEntry) {
  // T <branch> <from> <to> <count>
  // Trace: {branch, from, to}
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("T 4b196f 4b19e0 4b19ef 2\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.From, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].first.To, 0x4b19efULL);
  EXPECT_EQ(Traces[0].second.TakenCount, 2u);
}

TEST_F(PreAggregatedX86TestHelper, ReturnEntry) {
  // R <branch> <from> <to> <count>
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("R 4b196f 4b19e0 4b19ef 4\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.From, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].first.To, 0x4b19efULL);
  EXPECT_EQ(Traces[0].second.TakenCount, 4u);
}

TEST_F(PreAggregatedX86TestHelper, TraceWithNeg1AsBROnly) {
  // T entry with -1 as fall-through target: parsed as BR_ONLY.
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("T 4b196f 4b19e0 -1 2\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.Branch, 0x4b196fULL);
  EXPECT_EQ(Traces[0].first.From, 0x4b19e0ULL);
  EXPECT_EQ(Traces[0].first.To, Trace::BR_ONLY);
  EXPECT_EQ(Traces[0].second.TakenCount, 2u);
}

TEST_F(PreAggregatedX86TestHelper, TraceWithFFFAsBROnly) {
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("T 4b196f 4b19e0 ffffffffffffffff 2\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.To, Trace::BR_ONLY);
}

TEST_F(PreAggregatedX86TestHelper, TraceWithBuildIdNeg1) {
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("T 4b196f 4b19e0 deadbeef:-1 2\n", Traces);
  ASSERT_EQ(Traces.size(), 1u);
  EXPECT_EQ(Traces[0].first.To, Trace::BR_ONLY);
}

TEST_F(PreAggregatedX86TestHelper, MultipleEntries) {
  std::vector<std::pair<Trace, TakenBranchInfo>> Traces;
  parseAndCollectTraces("B 100 200 1 0\n"
                        "F 300 400 2\n"
                        "f 500 600 3\n"
                        "r 700 800 4\n"
                        "T 900 a00 b00 5\n",
                        Traces);
  ASSERT_EQ(Traces.size(), 5u);
  EXPECT_EQ(Traces[0].first.To, Trace::BR_ONLY);
  EXPECT_EQ(Traces[0].second.TakenCount, 1u);
  EXPECT_EQ(Traces[1].first.Branch, Trace::FT_ONLY);
  EXPECT_EQ(Traces[1].second.TakenCount, 2u);
  EXPECT_EQ(Traces[2].first.Branch, Trace::FT_EXTERNAL_ORIGIN);
  EXPECT_EQ(Traces[2].second.TakenCount, 3u);
  EXPECT_EQ(Traces[3].first.Branch, Trace::FT_EXTERNAL_RETURN);
  EXPECT_EQ(Traces[3].second.TakenCount, 4u);
  EXPECT_EQ(Traces[4].first.Branch, 0x900ULL);
  EXPECT_EQ(Traces[4].first.From, 0xa00ULL);
  EXPECT_EQ(Traces[4].first.To, 0xb00ULL);
  EXPECT_EQ(Traces[4].second.TakenCount, 5u);
}

#endif
