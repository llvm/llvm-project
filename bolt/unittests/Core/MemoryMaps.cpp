//===- bolt/unittest/Core/MemoryMaps.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryContext.h"
#include "bolt/Profile/DataAggregator.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

namespace opts {
extern cl::opt<std::string> ReadPerfEvents;
} // namespace opts

namespace {

/// Perform checks on memory map events normally captured in perf. Tests use
/// the 'opts::ReadPerfEvents' flag to emulate these events, passing a custom
/// 'perf script' output to DataAggregator.
struct MemoryMapsTester : public testing::TestWithParam<Triple::ArchType> {
  void SetUp() override {
    initalizeLLVM();
    prepareElf();
    initializeBOLT();
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

  void initializeBOLT() {
    Relocation::Arch = ObjFile->makeTriple().getArch();
    BC = cantFail(BinaryContext::createBinaryContext(
        ObjFile->makeTriple(), std::make_shared<orc::SymbolStringPool>(),
        ObjFile->getFileName(), nullptr, true, DWARFContext::create(*ObjFile),
        {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};
} // namespace

#ifdef X86_AVAILABLE

INSTANTIATE_TEST_SUITE_P(X86, MemoryMapsTester,
                         ::testing::Values(Triple::x86_64));

#endif

#ifdef AARCH64_AVAILABLE

INSTANTIATE_TEST_SUITE_P(AArch64, MemoryMapsTester,
                         ::testing::Values(Triple::aarch64));

#endif

/// Check that the correct mmap size is computed when we have multiple text
/// segment mappings.
TEST_P(MemoryMapsTester, ParseMultipleSegments) {
  const int Pid = 1234;
  StringRef Filename = "BINARY";
  opts::ReadPerfEvents = formatv(
      "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
      "[0xabc0000000(0x1000000) @ 0x11c0000 103:01 1573523 0]: r-xp {1}\n"
      "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
      "[0xabc2000000(0x8000000) @ 0x31d0000 103:01 1573523 0]: r-xp {1}\n",
      Pid, Filename);

  BC->SegmentMapInfo[0x11da000] = SegmentInfo{
      0x11da000, 0x10da000, 0x11ca000, 0x10da000, 0x10000, true, false};
  BC->SegmentMapInfo[0x31d0000] = SegmentInfo{
      0x31d0000, 0x51ac82c, 0x31d0000, 0x3000000, 0x200000, true, false};

  DataAggregator DA("");
  BC->setFilename(Filename);
  Error Err = DA.preprocessProfile(*BC);

  // Ignore errors from perf2bolt when parsing memory events later on.
  ASSERT_THAT_ERROR(std::move(Err), Succeeded());

  auto &BinaryMMapInfo = DA.getBinaryMMapInfo();
  auto El = BinaryMMapInfo.find(Pid);
  // Check that memory mapping is present and has the expected size.
  ASSERT_NE(El, BinaryMMapInfo.end());
  ASSERT_EQ(El->second.Size, static_cast<uint64_t>(0xb1d0000));
}

/// Check that DataAggregator aborts when pre-processing an input binary
/// with multiple text segments that have different base addresses.
TEST_P(MemoryMapsTester, MultipleSegmentsMismatchedBaseAddress) {
  const int Pid = 1234;
  StringRef Filename = "BINARY";
  opts::ReadPerfEvents = formatv(
      "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
      "[0xabc0000000(0x1000000) @ 0x11c0000 103:01 1573523 0]: r-xp {1}\n"
      "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
      "[0xabc2000000(0x8000000) @ 0x31d0000 103:01 1573523 0]: r-xp {1}\n",
      Pid, Filename);

  BC->SegmentMapInfo[0x11da000] = SegmentInfo{
      0x11da000, 0x10da000, 0x11ca000, 0x10da000, 0x10000, true, false};
  // Using '0x31d0fff' FileOffset which triggers a different base address
  // for this second text segment.
  BC->SegmentMapInfo[0x31d0000] = SegmentInfo{
      0x31d0000, 0x51ac82c, 0x31d0fff, 0x3000000, 0x200000, true, false};

  DataAggregator DA("");
  BC->setFilename(Filename);
  ASSERT_DEBUG_DEATH(
      { Error Err = DA.preprocessProfile(*BC); },
      "Base address on multiple segment mappings should match");
}
