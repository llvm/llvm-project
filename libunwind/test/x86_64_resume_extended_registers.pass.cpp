//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{x86_64-.+}}
// UNSUPPORTED: target={{.*-windows.*}}, msan

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__FreeBSD__)
#include <machine/sysarch.h>
#include <sys/syscall.h>
#elif defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define SYMBOL(x) STRINGIFY(__USER_LABEL_PREFIX__) #x

struct Snapshot {
  uint64_t flags;
  uint16_t selectors[6];
  uint64_t fsWord, gsWord;
  uint64_t oldFS, oldGS;
  uint64_t rcx;
};
static Snapshot snapshot;
static uint16_t expectedSelectors[6];
#if defined(__FreeBSD__) || defined(__linux__)
static uint64_t fsWord = 0x123456789abcdef0;
static uint64_t gsWord = 0xfedcba9876543210;
#endif
alignas(16) static char resumeStack[16384];

extern "C" void verify_resume() {
  assert(snapshot.flags & 1);
  for (unsigned i = 0; i != 6; ++i)
    assert(snapshot.selectors[i] == expectedSelectors[i]);
  assert(snapshot.rcx == 0xabcdef);
#if defined(__FreeBSD__) || defined(__linux__)
  assert(snapshot.fsWord == fsWord);
  assert(snapshot.gsWord == gsWord);
#endif
  _Exit(0);
}

// Record state before any flag-changing instruction or C call. Restore the
// original TLS bases with raw syscalls before entering C again.
extern "C" __attribute__((naked)) void resumed() {
  __asm__(
      "pushfq\n"
      "popq %rax\n"
      "movq %rax, 0(%r12)\n"
      "movw %es, 8(%r12)\n"
      "movw %cs, 10(%r12)\n"
      "movw %ss, 12(%r12)\n"
      "movw %ds, 14(%r12)\n"
      "movw %fs, 16(%r12)\n"
      "movw %gs, 18(%r12)\n"
      "movq %rcx, 56(%r12)\n"
#if defined(__FreeBSD__) || defined(__linux__)
      "movq %fs:0, %rax\n"
      "movq %rax, 24(%r12)\n"
      "movq %gs:0, %rax\n"
      "movq %rax, 32(%r12)\n"
#if defined(__FreeBSD__)
      "movq $" STRINGIFY(
          SYS_sysarch) ", %rax\n"
                       "movq $" STRINGIFY(
                           AMD64_SET_FSBASE) ", %rdi\n"
                                             "leaq 40(%r12), %rsi\n"
                                             "syscall\n"
                                             "jc 1f\n"
                                             "movq $" STRINGIFY(
                                                 SYS_sysarch) ", %rax\n"
                                                              "movq "
                                                              "$" STRINGIFY(
                                                                  AMD64_SET_GSBASE) ", %rdi\n"
                                                                                    "leaq 48(%r12), %rsi\n"
                                                                                    "syscall\n"
                                                                                    "jc 1f\n"
#else
      "movq $" STRINGIFY(
          SYS_arch_prctl) ", %rax\n"
                          "movq $" STRINGIFY(
                              ARCH_SET_FS) ", %rdi\n"
                                           "movq 40(%r12), %rsi\n"
                                           "syscall\n"
                                           "testq %rax, %rax\n"
                                           "js 1f\n"
                                           "movq $" STRINGIFY(
                                               SYS_arch_prctl) ", %rax\n"
                                                               "movq "
                                                               "$" STRINGIFY(
                                                                   ARCH_SET_GS) ", %rdi\n"
                                                                                "movq 48(%r12), %rsi\n"
                                                                                "syscall\n"
                                                                                "testq %rax, %rax\n"
                                                                                "js 1f\n"
#endif
#endif
                                                                                    "jmp " SYMBOL(
                                                                                        verify_resume) "\n"
                                                                                                       "1: ud2\n");
}

static unw_word_t get(unw_cursor_t &cursor, int reg) {
  unw_word_t value;
  assert(unw_get_reg(&cursor, reg, &value) == UNW_ESUCCESS);
  return value;
}

int main(int, char **) {
  unw_context_t context;
  unw_cursor_t cursor;
  assert(unw_getcontext(&context) == UNW_ESUCCESS);
  assert(unw_init_local(&cursor, &context) == UNW_ESUCCESS);
  const int selectors[] = {UNW_X86_64_ES, UNW_X86_64_CS, UNW_X86_64_SS,
                           UNW_X86_64_DS, UNW_X86_64_FS, UNW_X86_64_GS};
  for (unsigned i = 0; i != 6; ++i)
    expectedSelectors[i] = get(cursor, selectors[i]);
  // SS names a writable data segment and can also be loaded into DS and ES.
  expectedSelectors[0] = expectedSelectors[3] = expectedSelectors[2];
  assert(unw_set_reg(&cursor, UNW_X86_64_ES, expectedSelectors[0]) ==
         UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_X86_64_DS, expectedSelectors[3]) ==
         UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_X86_64_RFLAGS,
                     (get(cursor, UNW_X86_64_RFLAGS) & ~unw_word_t(0x400)) |
                         1) == UNW_ESUCCESS);
#if defined(__FreeBSD__) || defined(__linux__)
  snapshot.oldFS = get(cursor, UNW_X86_64_FS_BASE);
  snapshot.oldGS = get(cursor, UNW_X86_64_GS_BASE);
  assert(unw_set_reg(&cursor, UNW_X86_64_FS_BASE, uintptr_t(&fsWord)) ==
         UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_X86_64_GS_BASE, uintptr_t(&gsWord)) ==
         UNW_ESUCCESS);
#endif
  assert(unw_set_reg(&cursor, UNW_X86_64_R12, uintptr_t(&snapshot)) ==
         UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_X86_64_RCX, 0xabcdef) == UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_REG_SP,
                     uintptr_t(resumeStack + sizeof(resumeStack) - 8)) ==
         UNW_ESUCCESS);
  assert(unw_set_reg(&cursor, UNW_REG_IP, uintptr_t(&resumed)) == UNW_ESUCCESS);
  unw_resume(&cursor);
  abort();
}
