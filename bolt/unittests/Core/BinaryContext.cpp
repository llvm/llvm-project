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
        ObjFile.get(), true, DWARFContext::create(*ObjFile.get())));
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

  Optional<uint64_t> BaseAddress =
      BC->getBaseAddressForMapping(0x7f13f5556000, 0x10e8c000);
  ASSERT_TRUE(BaseAddress.hasValue());
  ASSERT_EQ(*BaseAddress, 0x7f13e46c9000ULL);

  BaseAddress = BC->getBaseAddressForMapping(0x7f13f5556000, 0x137a000);
  ASSERT_FALSE(BaseAddress.hasValue());
}
