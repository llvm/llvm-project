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

// FreeBSD may clear r8-r10 on syscall return. Context capture must retain its
// destination across both base queries and preserve any callee-saved register
// used for that purpose. Set RBX in assembly so the compiler cannot choose a
// different register or hide a clobber by reloading its value.
extern "C" __attribute__((naked)) int capture_with_rbx(unw_context_t *,
                                                       uint64_t *, uint64_t) {
  // clang-format off
  __asm__("pushq %rbx\n"
          "pushq %r12\n"
          "subq $8, %rsp\n" // Align the stack before calling unw_getcontext.
          "movq %rsi, %r12\n"
          "movq %rdx, %rbx\n"
          "call " SYMBOL(unw_getcontext) "\n"
          "movq %rbx, (%r12)\n"
          "addq $8, %rsp\n"
          "popq %r12\n"
          "popq %rbx\n"
          "ret\n");
  // clang-format on
}

static unw_word_t get(unw_cursor_t &cursor, int reg) {
  unw_word_t value;
  assert(unw_get_reg(&cursor, reg, &value) == UNW_ESUCCESS);
  return value;
}

int main(int, char **) {
  const uint64_t expected = UINT64_C(0x123456789abcdef0);
  unw_context_t context;
  uint64_t actual = 0;
  assert(capture_with_rbx(&context, &actual, expected) == UNW_ESUCCESS);
  assert(actual == expected);

  unw_cursor_t cursor;
  assert(unw_init_local(&cursor, &context) == UNW_ESUCCESS);
  assert(get(cursor, UNW_X86_64_RBX) == expected);

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
#endif
  return 0;
}
