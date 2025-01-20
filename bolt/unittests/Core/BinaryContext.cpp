//===- bolt/unittest/Core/BinaryContext.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryContext.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

namespace {
struct BinaryContextTester : public testing::TestWithParam<Triple::ArchType> {
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
        ObjFile->getFileName(), nullptr, true,
        DWARFContext::create(*ObjFile.get()), {llvm::outs(), llvm::errs()}));
    ASSERT_FALSE(!BC);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};
} // namespace

#ifdef X86_AVAILABLE

INSTANTIATE_TEST_SUITE_P(X86, BinaryContextTester,
                         ::testing::Values(Triple::x86_64));

#endif

#ifdef AARCH64_AVAILABLE

INSTANTIATE_TEST_SUITE_P(AArch64, BinaryContextTester,
                         ::testing::Values(Triple::aarch64));

TEST_P(BinaryContextTester, FlushPendingRelocCALL26) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // This test checks that encodeValueAArch64 used by flushPendingRelocations
  // returns correctly encoded values for CALL26 relocation for both backward
  // and forward branches.
  //
  // The offsets layout is:
  // 4:  func1
  // 8:  bl func1
  // 12: bl func2
  // 16: func2

  constexpr size_t DataSize = 20;
  uint8_t *Data = new uint8_t[DataSize];
  BinarySection &BS = BC->registerOrUpdateSection(
      ".text", ELF::SHT_PROGBITS, ELF::SHF_EXECINSTR | ELF::SHF_ALLOC, Data,
      DataSize, 4);
  MCSymbol *RelSymbol1 = BC->getOrCreateGlobalSymbol(4, "Func1");
  ASSERT_TRUE(RelSymbol1);
  BS.addPendingRelocation(
      Relocation{8, RelSymbol1, ELF::R_AARCH64_CALL26, 0, 0});
  MCSymbol *RelSymbol2 = BC->getOrCreateGlobalSymbol(16, "Func2");
  ASSERT_TRUE(RelSymbol2);
  BS.addPendingRelocation(
      Relocation{12, RelSymbol2, ELF::R_AARCH64_CALL26, 0, 0});

  SmallVector<char> Vect(DataSize);
  raw_svector_ostream OS(Vect);

  BS.flushPendingRelocations(OS, [&](const MCSymbol *S) {
    return S == RelSymbol1 ? 4 : S == RelSymbol2 ? 16 : 0;
  });

  const uint8_t Func1Call[4] = {255, 255, 255, 151};
  const uint8_t Func2Call[4] = {1, 0, 0, 148};

  EXPECT_FALSE(memcmp(Func1Call, &Vect[8], 4)) << "Wrong backward call value\n";
  EXPECT_FALSE(memcmp(Func2Call, &Vect[12], 4)) << "Wrong forward call value\n";
}

TEST_P(BinaryContextTester, FlushPendingRelocJUMP26) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // This test checks that encodeValueAArch64 used by flushPendingRelocations
  // returns correctly encoded values for R_AARCH64_JUMP26 relocation for both
  // backward and forward branches.
  //
  // The offsets layout is:
  // 4:  func1
  // 8:  b func1
  // 12: b func2
  // 16: func2

  const uint64_t Size = 20;
  char *Data = new char[Size];
  BinarySection &BS = BC->registerOrUpdateSection(
      ".text", ELF::SHT_PROGBITS, ELF::SHF_EXECINSTR | ELF::SHF_ALLOC,
      (uint8_t *)Data, Size, 4);
  MCSymbol *RelSymbol1 = BC->getOrCreateGlobalSymbol(4, "Func1");
  ASSERT_TRUE(RelSymbol1);
  BS.addPendingRelocation(
      Relocation{8, RelSymbol1, ELF::R_AARCH64_JUMP26, 0, 0});
  MCSymbol *RelSymbol2 = BC->getOrCreateGlobalSymbol(16, "Func2");
  ASSERT_TRUE(RelSymbol2);
  BS.addPendingRelocation(
      Relocation{12, RelSymbol2, ELF::R_AARCH64_JUMP26, 0, 0});

  SmallVector<char> Vect(Size);
  raw_svector_ostream OS(Vect);

  BS.flushPendingRelocations(OS, [&](const MCSymbol *S) {
    return S == RelSymbol1 ? 4 : S == RelSymbol2 ? 16 : 0;
  });

  const uint8_t Func1Call[4] = {255, 255, 255, 23};
  const uint8_t Func2Call[4] = {1, 0, 0, 20};

  EXPECT_FALSE(memcmp(Func1Call, &Vect[8], 4))
      << "Wrong backward branch value\n";
  EXPECT_FALSE(memcmp(Func2Call, &Vect[12], 4))
      << "Wrong forward branch value\n";
}

TEST_P(BinaryContextTester, FlushOptionalOutOfRangePendingRelocCALL26) {
  if (GetParam() != Triple::aarch64)
    GTEST_SKIP();

  // This test checks that flushPendingRelocations skips flushing any optional
  // pending relocations that cannot be encoded.

  bool DebugFlagPrev = ::llvm::DebugFlag;
  ::llvm::DebugFlag = true;
  testing::internal::CaptureStderr();

  BinarySection &BS = BC->registerOrUpdateSection(
      ".text", ELF::SHT_PROGBITS, ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
  MCSymbol *RelSymbol = BC->getOrCreateGlobalSymbol(4, "Func");
  ASSERT_TRUE(RelSymbol);
  BS.addPendingRelocation(Relocation{8, RelSymbol, ELF::R_AARCH64_CALL26, 0, 0},
                          /*Optional*/ true);

  SmallVector<char> Vect;
  raw_svector_ostream OS(Vect);

  // Resolve relocation symbol to a high value so encoding will be out of range.
  BS.flushPendingRelocations(OS, [&](const MCSymbol *S) { return 0x800000F; });

  std::string CapturedStdErr = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(CapturedStdErr.find("BOLT-INFO: Skipped 1 pending relocations as "
                                  "they were out of range") !=
              std::string::npos);

  ::llvm::DebugFlag = DebugFlagPrev;
}

#endif

TEST_P(BinaryContextTester, BaseAddress) {
  // Check that  base address calculation is correct for a binary with the
  // following segment layout:
  BC->SegmentMapInfo[0] =
      SegmentInfo{0, 0x10e8c2b4, 0, 0x10e8c2b4, 0x1000, true};
  BC->SegmentMapInfo[0x10e8d2b4] =
      SegmentInfo{0x10e8d2b4, 0x3952faec, 0x10e8c2b4, 0x3952faec, 0x1000, true};
  BC->SegmentMapInfo[0x4a3bddc0] =
      SegmentInfo{0x4a3bddc0, 0x148e828, 0x4a3bbdc0, 0x148e828, 0x1000, true};
  BC->SegmentMapInfo[0x4b84d5e8] =
      SegmentInfo{0x4b84d5e8, 0x294f830, 0x4b84a5e8, 0x3d3820, 0x1000, true};

  std::optional<uint64_t> BaseAddress =
      BC->getBaseAddressForMapping(0x7f13f5556000, 0x10e8c000);
  ASSERT_TRUE(BaseAddress.has_value());
  ASSERT_EQ(*BaseAddress, 0x7f13e46c9000ULL);

  BaseAddress = BC->getBaseAddressForMapping(0x7f13f5556000, 0x137a000);
  ASSERT_FALSE(BaseAddress.has_value());
}

TEST_P(BinaryContextTester, BaseAddress2) {
  // Check that base address calculation is correct for a binary if the
  // alignment in ELF file are different from pagesize.
  // The segment layout is as follows:
  BC->SegmentMapInfo[0] = SegmentInfo{0, 0x2177c, 0, 0x2177c, 0x10000, true};
  BC->SegmentMapInfo[0x31860] =
      SegmentInfo{0x31860, 0x370, 0x21860, 0x370, 0x10000, true};
  BC->SegmentMapInfo[0x41c20] =
      SegmentInfo{0x41c20, 0x1f8, 0x21c20, 0x1f8, 0x10000, true};
  BC->SegmentMapInfo[0x54e18] =
      SegmentInfo{0x54e18, 0x51, 0x24e18, 0x51, 0x10000, true};

  std::optional<uint64_t> BaseAddress =
      BC->getBaseAddressForMapping(0xaaaaea444000, 0x21000);
  ASSERT_TRUE(BaseAddress.has_value());
  ASSERT_EQ(*BaseAddress, 0xaaaaea413000ULL);

  BaseAddress = BC->getBaseAddressForMapping(0xaaaaea444000, 0x11000);
  ASSERT_FALSE(BaseAddress.has_value());
}

TEST_P(BinaryContextTester, BaseAddressSegmentsSmallerThanAlignment) {
  // Check that the correct segment is used to compute the base address
  // when multiple segments are close together in the ELF file (closer
  // than the required alignment in the process space).
  // See https://github.com/llvm/llvm-project/issues/109384
  BC->SegmentMapInfo[0] = SegmentInfo{0, 0x1d1c, 0, 0x1d1c, 0x10000, false};
  BC->SegmentMapInfo[0x11d40] =
      SegmentInfo{0x11d40, 0x11e0, 0x1d40, 0x11e0, 0x10000, true};
  BC->SegmentMapInfo[0x22f20] =
      SegmentInfo{0x22f20, 0x10e0, 0x2f20, 0x1f0, 0x10000, false};
  BC->SegmentMapInfo[0x33110] =
      SegmentInfo{0x33110, 0x89, 0x3110, 0x88, 0x10000, false};

  std::optional<uint64_t> BaseAddress =
      BC->getBaseAddressForMapping(0xaaaaaaab1000, 0x1000);
  ASSERT_TRUE(BaseAddress.has_value());
  ASSERT_EQ(*BaseAddress, 0xaaaaaaaa0000ULL);
}
