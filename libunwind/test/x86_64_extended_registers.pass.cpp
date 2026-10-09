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
#include <string.h>

#if defined(__FreeBSD__)
#include <machine/sysarch.h>
#elif defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define SYMBOL(x) STRINGIFY(__USER_LABEL_PREFIX__) #x

static const int selectors[] = {UNW_X86_64_ES, UNW_X86_64_CS, UNW_X86_64_SS,
                                UNW_X86_64_DS, UNW_X86_64_FS, UNW_X86_64_GS};
static const char *names[] = {"es", "cs", "ss", "ds", "fs", "gs"};

static unw_word_t get(unw_cursor_t &cursor, int reg) {
  unw_word_t value;
  assert(unw_get_reg(&cursor, reg, &value) == UNW_ESUCCESS);
  return value;
}

// A tail call preserves the flags and stack pointer of our caller. In
// particular, getcontext must capture CF before adjusting its saved SP.
extern "C" __attribute__((naked)) int capture(unw_context_t *) {
  __asm__("stc\n"
          "jmp " SYMBOL(unw_getcontext) "\n");
}

static void test_capture_and_access() {
  unw_context_t context;
  // Resolve the shared-library entry point before testing CF: lazy binding
  // can change flags between capture's tail call and getcontext's entry.
  assert(unw_getcontext(&context) == UNW_ESUCCESS);
  memset(&context, 0xa5, sizeof(context));
  assert(capture(&context) == UNW_ESUCCESS);
  unw_cursor_t cursor;
  assert(unw_init_local(&cursor, &context) == UNW_ESUCCESS);
  assert(get(cursor, UNW_X86_64_RFLAGS) & 1);
  assert(strcmp(unw_regname(&cursor, UNW_X86_64_RFLAGS), "rflags") == 0);

  uint16_t actual[6];
  __asm__ volatile("movw %%es, %0\n"
                   "movw %%cs, %1\n"
                   "movw %%ss, %2\n"
                   "movw %%ds, %3\n"
                   "movw %%fs, %4\n"
                   "movw %%gs, %5\n"
                   : "=m"(actual[0]), "=m"(actual[1]), "=m"(actual[2]),
                     "=m"(actual[3]), "=m"(actual[4]), "=m"(actual[5]));
  for (unsigned i = 0; i != 6; ++i) {
    assert(get(cursor, selectors[i]) == actual[i]);
    assert(strcmp(unw_regname(&cursor, selectors[i]), names[i]) == 0);
    assert(unw_set_reg(&cursor, selectors[i], 0x12345678) == UNW_ESUCCESS);
    assert(get(cursor, selectors[i]) == 0x5678);
  }

#if defined(__FreeBSD__) || defined(__linux__)
  uintptr_t fs, gs;
#if defined(__FreeBSD__)
  assert(sysarch(AMD64_GET_FSBASE, &fs) == 0);
  assert(sysarch(AMD64_GET_GSBASE, &gs) == 0);
#else
  assert(syscall(SYS_arch_prctl, ARCH_GET_FS, &fs) == 0);
  assert(syscall(SYS_arch_prctl, ARCH_GET_GS, &gs) == 0);
#endif
  assert(get(cursor, UNW_X86_64_FS_BASE) == fs);
  assert(get(cursor, UNW_X86_64_GS_BASE) == gs);
  assert(strcmp(unw_regname(&cursor, UNW_X86_64_FS_BASE), "fs.base") == 0);
  assert(strcmp(unw_regname(&cursor, UNW_X86_64_GS_BASE), "gs.base") == 0);
  assert(unw_set_reg(&cursor, UNW_X86_64_FS_BASE, 0x12345678) == UNW_ESUCCESS);
  assert(get(cursor, UNW_X86_64_FS_BASE) == 0x12345678);
  assert(unw_set_reg(&cursor, UNW_X86_64_GS_BASE, 0x87654321) == UNW_ESUCCESS);
  assert(get(cursor, UNW_X86_64_GS_BASE) == 0x87654321);
#endif

  unw_word_t value;
  assert(unw_get_reg(&cursor, 56, &value) == UNW_EBADREG);
  assert(unw_get_reg(&cursor, 57, &value) == UNW_EBADREG);
  assert(unw_get_reg(&cursor, 60, &value) == UNW_EBADREG);
}

extern "C" void check_cfi() {
  unw_context_t context;
  unw_cursor_t cursor;
  assert(unw_getcontext(&context) == UNW_ESUCCESS);
  assert(unw_init_local(&cursor, &context) == UNW_ESUCCESS);
  assert(unw_step(&cursor) > 0); // to the synthetic trampoline
  assert(unw_step(&cursor) > 0); // recover its saved registers
  assert(get(cursor, UNW_X86_64_RFLAGS) == 0x203);
  for (unsigned i = 0; i != 6; ++i)
    assert(get(cursor, selectors[i]) == unsigned(selectors[i]));
#if defined(__FreeBSD__) || defined(__linux__)
  assert(get(cursor, UNW_X86_64_FS_BASE) == 0x12345678);
  assert(get(cursor, UNW_X86_64_GS_BASE) == 0x87654321);
#endif
}

// Model the previously disabled FreeBSD sigtramp CFI, including the packed
// 16-bit selectors whose adjacent data must not become part of their values.
__attribute__((naked)) void trampoline() {
  __asm__(".cfi_signal_frame\n"
          "pushq %rbp\n"
          ".cfi_def_cfa_offset 16\n"
          ".cfi_offset 6, -16\n"
          "movq %rsp, %rbp\n"
          ".cfi_def_cfa_register 6\n"
          "subq $80, %rsp\n"
          "movq $0x203, -8(%rbp)\n"
          ".cfi_offset 49, -24\n"
          "movq $50, -16(%rbp)\n"
          "movw $0xabcd, -14(%rbp)\n"
          ".cfi_offset 50, -32\n"
          "movq $51, -24(%rbp)\n"
          ".cfi_offset 51, -40\n"
          "movq $52, -32(%rbp)\n"
          ".cfi_offset 52, -48\n"
          "movq $53, -40(%rbp)\n"
          "movw $0xabcd, -38(%rbp)\n"
          ".cfi_offset 53, -56\n"
          "movq $54, -48(%rbp)\n"
          "movw $0xabcd, -46(%rbp)\n"
          ".cfi_offset 54, -64\n"
          "movq $55, -56(%rbp)\n"
          "movw $0xabcd, -54(%rbp)\n"
          ".cfi_offset 55, -72\n"
#if defined(__FreeBSD__) || defined(__linux__)
          "movq $0x12345678, -64(%rbp)\n"
          ".cfi_offset 58, -80\n"
          "movl $0x87654321, -72(%rbp)\n"
          "movl $0, -68(%rbp)\n"
          ".cfi_offset 59, -88\n"
#endif
          "call " SYMBOL(check_cfi) "\n"
                                    "leave\n"
                                    ".cfi_def_cfa 7, 8\n"
                                    ".cfi_restore 6\n"
                                    "ret\n");
}

int main(int, char **) {
  test_capture_and_access();
  trampoline();
  return 0;
}
