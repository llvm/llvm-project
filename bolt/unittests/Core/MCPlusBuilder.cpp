#ifdef AARCH64_AVAILABLE
#include "AArch64Subtarget.h"
#endif // AARCH64_AVAILABLE

#ifdef X86_AVAILABLE
#include "X86Subtarget.h"
#endif // X86_AVAILABLE

#include "bolt/Rewrite/RewriteInstance.h"
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
struct MCPlusBuilderTester : public testing::TestWithParam<Triple::ArchType> {
  void SetUp() override {
    initalizeLLVM();
    prepareElf();
    initializeBolt();
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

  void initializeBolt() {
    BC = BinaryContext::createBinaryContext(
        ObjFile.get(), true, DWARFContext::create(*ObjFile.get()));
    ASSERT_FALSE(!BC);
    BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(createMCPlusBuilder(
        GetParam(), BC->MIA.get(), BC->MII.get(), BC->MRI.get())));
  }

  void testRegAliases(Triple::ArchType Arch, uint64_t Register,
                      uint64_t *Aliases, size_t Count,
                      bool OnlySmaller = false) {
    if (GetParam() != Arch)
      GTEST_SKIP();

    const BitVector &BV = BC->MIB->getAliases(Register, OnlySmaller);
    ASSERT_EQ(BV.count(), Count);
    for (size_t I = 0; I < Count; ++I)
      ASSERT_TRUE(BV[Aliases[I]]);
  }

  char ElfBuf[sizeof(typename ELF64LE::Ehdr)] = {};
  std::unique_ptr<ObjectFile> ObjFile;
  std::unique_ptr<BinaryContext> BC;
};
} // namespace

#ifdef AARCH64_AVAILABLE

INSTANTIATE_TEST_SUITE_P(AArch64, MCPlusBuilderTester,
                         ::testing::Values(Triple::aarch64));

TEST_P(MCPlusBuilderTester, AliasX0) {
  uint64_t AliasesX0[] = {AArch64::W0, AArch64::X0, AArch64::W0_W1,
                          AArch64::X0_X1, AArch64::X0_X1_X2_X3_X4_X5_X6_X7};
  size_t AliasesX0Count = sizeof(AliasesX0) / sizeof(*AliasesX0);
  testRegAliases(Triple::aarch64, AArch64::X0, AliasesX0, AliasesX0Count);
}

TEST_P(MCPlusBuilderTester, AliasSmallerX0) {
  uint64_t AliasesX0[] = {AArch64::W0, AArch64::X0};
  size_t AliasesX0Count = sizeof(AliasesX0) / sizeof(*AliasesX0);
  testRegAliases(Triple::aarch64, AArch64::X0, AliasesX0, AliasesX0Count, true);
}

#endif // AARCH64_AVAILABLE

#ifdef X86_AVAILABLE

INSTANTIATE_TEST_SUITE_P(X86, MCPlusBuilderTester,
                         ::testing::Values(Triple::x86_64));

TEST_P(MCPlusBuilderTester, AliasAX) {
  uint64_t AliasesAX[] = {X86::RAX, X86::EAX, X86::AX, X86::AL, X86::AH};
  size_t AliasesAXCount = sizeof(AliasesAX) / sizeof(*AliasesAX);
  testRegAliases(Triple::x86_64, X86::AX, AliasesAX, AliasesAXCount);
}

TEST_P(MCPlusBuilderTester, AliasSmallerAX) {
  uint64_t AliasesAX[] = {X86::AX, X86::AL, X86::AH};
  size_t AliasesAXCount = sizeof(AliasesAX) / sizeof(*AliasesAX);
  testRegAliases(Triple::x86_64, X86::AX, AliasesAX, AliasesAXCount, true);
}

#endif // X86_AVAILABLE
