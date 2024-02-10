#include "bolt/Rewrite/JITRewriteInstance.h"
#include "bolt/Core/BinaryContext.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace bolt;

namespace {
struct JITRewriteInstanceTester
    : public testing::TestWithParam<Triple::ArchType> {
  void SetUp() override {
    initalizeLLVM();
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

  void initializeBOLT() {
    BOLTJIT = cantFail(bolt::JITRewriteInstance::createJITRewriteInstance(
        {llvm::outs(), llvm::errs()}, /*IsPIC*/ false));
    ASSERT_FALSE(!BOLTJIT);
  }

  std::unique_ptr<JITRewriteInstance> BOLTJIT;
};
} // namespace

#ifdef X86_AVAILABLE

// clang-format off
extern "C" __attribute((naked)) int fib(int n)
{
  __asm__ __volatile__(
    "pushq   %%r14\n"
    "pushq   %%rbx\n"
    "pushq   %%rax\n"
    "movl    %%edi, %%r14d\n"
    "xorl    %%ebx, %%ebx\n"
    "cmpl    $0x2, %%edi\n"
    "jge     .Ltmp0\n"
    "movl    %%r14d, %%ecx\n"
    "jmp     .Ltmp1\n"
    ".Ltmp0:\n"
    "xorl    %%ebx, %%ebx\n"
    "nopw    %%cs:(%%rax,%%rax)\n"
    ".Ltmp2:\n"
    "leal    -0x1(%%r14), %%edi\n"
    "callq   fib\n"
    "leal    -0x2(%%r14), %%ecx\n"
    "addl    %%eax, %%ebx\n"
    "cmpl    $0x3, %%r14d\n"
    "movl    %%ecx, %%r14d\n"
    "ja      .Ltmp2\n"
    ".Ltmp1:\n"
    "addl    %%ecx, %%ebx\n"
    "movl    %%ebx, %%eax\n"
    "addq    $0x8, %%rsp\n"
    "popq    %%rbx\n"
    "popq    %%r14\n"
    "retq\n"
    :::);
}
// clang-format on

INSTANTIATE_TEST_SUITE_P(X86, JITRewriteInstanceTester,
                         ::testing::Values(Triple::x86_64));

TEST_P(JITRewriteInstanceTester, DisassembleFib) {
  EXPECT_EQ(fib(7), 13);

  // BOLT JIT test/example
  // Analyze fib function in this binary
  // Disassemble 63 bytes
  uint64_t Address = reinterpret_cast<uint64_t>(&fib);
  StringRef Data = StringRef(reinterpret_cast<const char *>(&fib), 63);

  BOLTJIT->registerJITSection(StringRef(".text.example"), Address, Data, 1,
                              ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC | ELF::SHF_EXECINSTR);
  BOLTJIT->registerJITFunction(StringRef("fib"), Address, 63);
  ASSERT_FALSE(BOLTJIT->run());

  // Print to screen
  BOLTJIT->printAll(outs());
}

#endif
