//===- bolt/unittests/Profile/PerfScripts.cpp--------------------------===//
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
#include <gmock/gmock.h>

using namespace llvm;
using namespace llvm::bolt;
using namespace llvm::object;
using namespace llvm::ELF;

namespace opts {
extern cl::opt<bool> ReadPreAggregated;
extern cl::opt<bool> ArmSPE;
} // namespace opts

namespace llvm {
namespace bolt {

/// Tests textual profile parsing using dummy input and
/// performs negative checks on PERFTEXT headers.
struct PerfScriptTestHelper : public testing::Test {
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
        DWARFContext::create(*ObjFile), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;

  void createTempFileWithContent(const std::string &Buffer,
                                 SmallVector<char, 256> &Path) {
    int FD;
    sys::fs::createTemporaryFile("perf-script", "text", FD, Path);
    ASSERT_GE(FD, 0);

    llvm::raw_fd_ostream FileStream(FD, true);
    FileStream << Buffer;
    FileStream.flush();
  }

  // Checks several type of parsing errors on pre-parsed file header.
  void checkPreParsedFileHeaderErrors(const std::string &Buffer,
                                      const std::string &ErrorMessage) {
    testing::internal::CaptureStderr();
    SmallVector<char, 256> Path{};
    createTempFileWithContent(Buffer, Path);

    DataAggregator DA(Path.data());

    DA.ParsingBuf = Buffer;
    DA.BC = BC.get();

    std::error_code EC = DA.parsePerfScriptFileHeader();
    ASSERT_TRUE(EC == llvm::errc::io_error);

    errs().flush();
    std::string CapturedStderr = testing::internal::GetCapturedStderr();
    EXPECT_THAT(CapturedStderr, testing::HasSubstr(ErrorMessage));

    sys::fs::remove(Path);
  }

  // Sanity check whether MAIN events are processed
  void parseAndCheckPerfScriptProfile(const std::string &Buffer, const int Pid,
                                      const size_t Expected) {
    SmallVector<char, 256> Path{};
    createTempFileWithContent(Buffer, Path);

    DataAggregator DA(Path.data());
    DA.BC = BC.get();
    DataAggregator::MMapInfo MMap;
    DA.BinaryMMapInfo.insert(std::make_pair(Pid, MMap));

    Error Err = DA.parsePerfScript();
    EXPECT_THAT_ERROR(std::move(Err), Succeeded());
    EXPECT_EQ(DA.Traces.size(), Expected);

    sys::fs::remove(Path);
  }
};

} // namespace bolt
} // namespace llvm

TEST_F(PerfScriptTestHelper, CheckMissingEndOfLineChar) {
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "expected rest of line";
  std::string Buffer =
      "PERFTEXT;BUILDIDS=32;MMAP=2DC6C0;MAIN=1388;TASK=55730;MEM=128;";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, CheckMissingPerfMagicString) {
  // Checks missing/wrong "PERFTEXT" string.
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "expected 'PERFTEXT' magic string";
  std::string Buffer =
      "PERF;BUILDIDS=32;MMAP=2DC6C0;MAIN=1388;TASK=55730;MEM=128;\n";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, CheckMissingEventAndSizeContent) {
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "expected type=length content";
  std::string Buffer = "PERFTEXT;BUILDID?1;\n";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, CheckMalformedTypes) {
  // Checks malformed type: actual: BUID, expected: BUILDID.
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "supported types: BUILDID, MAIN, MMAP, TASK, MEM";
  std::string Buffer =
      "PERFTEXT;BUID=32;MMAP=2DC6C0;MAIN=1388;TASK=55730;MEM=128;\n";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, CheckExpectedHexNumber) {
  // Checks expected hexadecimal number error message: BUILDIDS=32y.
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "expected hexadecimal number";
  std::string Buffer =
      "PERFTEXT;BUILDIDS=32y;MMAP=2DC6C0;MAIN=1388;TASK=55730;MEM=128;\n";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, CheckCorruptedTextProfile) {
  // Checks the sum of events length is not equal to file size.
  opts::ReadPreAggregated = true;
  std::string ErrorMessage = "corrupted perfscript profile";
  std::string Buffer =
      "PERFTEXT;BUILDIDS=32;MMAP=2DC6C0;MAIN=1388;TASK=55730;MEM=128;\n";

  checkPreParsedFileHeaderErrors(Buffer, ErrorMessage);
}

TEST_F(PerfScriptTestHelper, ParseAndCheckFileHeader) {
  opts::ReadPreAggregated = true;
  opts::ArmSPE = true;
  const int Pid = 1234;
  StringRef Filename = "ELF";
  std::string BuildID = formatv("{0} /example/{1}\n", Pid, Filename).str();
  std::string MainEvents = formatv("  {0}  0xa002/0xa003/PN/-/-/10/COND/-\n"
                                   "  {0}  0xb002/0xb003/P/-/-/4/RET/-\n"
                                   "  {0}  0xc456/0xc789/P/-/-/13/-/-\n",
                                   Pid)
                               .str();
  std::string MemEvents =
      formatv(
          "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
          "[0xabc0000000(0x1000000) @ 0x11c0000 103:01 1573523 0]: r-xp {1}\n"
          "name       0 [000]     0.000000: PERF_RECORD_MMAP2 {0}/{0}: "
          "[0xabc2000000(0x8000000) @ 0x31d0000 103:01 1573523 0]: r-xp {1}\n",
          Pid, Filename)
          .str();
  std::string TaskEvents =
      formatv("{1}   {0} PERF_RECORD_COMM exec: {1}:{0}/{0}\n"
              "{1}   {0} PERF_RECORD_EXIT({0}:{0}}):(20469:20469)\n",
              Pid, Filename)
          .str();

  std::string Header =
      formatv(
          "PERFTEXT;BUILDIDS={0:x-};MMAP={1:x-};MAIN={2:x-};TASK={3:x-};{4}\n",
          BuildID.size(), MemEvents.size(), MainEvents.size(),
          TaskEvents.size(), "   ")
          .str();

  std::string Buffer = formatv("{0}{1}{2}{3}{4}", Header, BuildID, MemEvents,
                               MainEvents, TaskEvents)
                           .str();

  // Defined 3 entries on MainEvents. The size of Traces intermediate storage
  // should be 'size == 3' after the parsing this dummy MainEvents.
  parseAndCheckPerfScriptProfile(Buffer, Pid, 3);
}
