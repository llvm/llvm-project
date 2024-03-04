#include "bolt/Core/BinaryContext.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ELFObjectFile.h"
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

  void initializeBOLT() {
    BC = cantFail(BinaryContext::createBinaryContext(
        ObjFile.get(), true, DWARFContext::create(*ObjFile.get()),
        {llvm::outs(), llvm::errs()}));
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
  BS.addRelocation(8, RelSymbol1, ELF::R_AARCH64_CALL26, 0, 0, true);
  MCSymbol *RelSymbol2 = BC->getOrCreateGlobalSymbol(16, "Func2");
  ASSERT_TRUE(RelSymbol2);
  BS.addRelocation(12, RelSymbol2, ELF::R_AARCH64_CALL26, 0, 0, true);

  std::error_code EC;
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

#endif

TEST_P(BinaryContextTester, BaseAddress) {
  // Check that  base address calculation is correct for a binary with the
  // following segment layout:
  BC->SegmentMapInfo[0] = SegmentInfo{0, 0x10e8c2b4, 0, 0x10e8c2b4, 0x1000};
  BC->SegmentMapInfo[0x10e8d2b4] =
      SegmentInfo{0x10e8d2b4, 0x3952faec, 0x10e8c2b4, 0x3952faec, 0x1000};
  BC->SegmentMapInfo[0x4a3bddc0] =
      SegmentInfo{0x4a3bddc0, 0x148e828, 0x4a3bbdc0, 0x148e828, 0x1000};
  BC->SegmentMapInfo[0x4b84d5e8] =
      SegmentInfo{0x4b84d5e8, 0x294f830, 0x4b84a5e8, 0x3d3820, 0x1000};

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
  BC->SegmentMapInfo[0] = SegmentInfo{0, 0x2177c, 0, 0x2177c, 0x10000};
  BC->SegmentMapInfo[0x31860] =
      SegmentInfo{0x31860, 0x370, 0x21860, 0x370, 0x10000};
  BC->SegmentMapInfo[0x41c20] =
      SegmentInfo{0x41c20, 0x1f8, 0x21c20, 0x1f8, 0x10000};
  BC->SegmentMapInfo[0x54e18] =
      SegmentInfo{0x54e18, 0x51, 0x24e18, 0x51, 0x10000};

  std::optional<uint64_t> BaseAddress =
      BC->getBaseAddressForMapping(0xaaaaea444000, 0x21000);
  ASSERT_TRUE(BaseAddress.has_value());
  ASSERT_EQ(*BaseAddress, 0xaaaaea413000ULL);

  BaseAddress = BC->getBaseAddressForMapping(0xaaaaea444000, 0x11000);
  ASSERT_FALSE(BaseAddress.has_value());
}
