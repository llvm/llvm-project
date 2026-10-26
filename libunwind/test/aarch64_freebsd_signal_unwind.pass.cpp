//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{aarch64-.*-freebsd.*}}
// UNSUPPORTED: msan

// Unwind through the kernel's signal trampoline, including on an alternate
// stack. In particular, restore the interrupted PC rather than the saved LR:
// the latter would skip the interrupted leaf function altogether.

#undef NDEBUG
#include "support/func_bounds.h"
#include <assert.h>
#include <libunwind.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ucontext.h>

FUNC_BOUNDS_DECL(main_func);
static volatile sig_atomic_t handled;

static void check_reg(unw_cursor_t *cursor, int reg, unw_word_t expected) {
  unw_word_t value;
  assert(unw_get_reg(cursor, reg, &value) == 0);
  assert(value == expected);
}

static void handler(int, siginfo_t *, void *context) {
  auto *uc = static_cast<ucontext_t *>(context);
  const mcontext_t &mc = uc->uc_mcontext;
  unw_context_t unwind_context;
  unw_cursor_t cursor;
  assert(unw_getcontext(&unwind_context) == 0);
  assert(unw_init_local(&cursor, &unwind_context) == 0);

  bool found = false;
  for (int i = 0; i < 64 && unw_step(&cursor) > 0; ++i) {
    if (unw_is_signal_frame(&cursor)) {
      found = true;
      break;
    }
  }
  assert(found);
  check_reg(&cursor, UNW_REG_IP, mc.mc_gpregs.gp_elr);
  check_reg(&cursor, UNW_REG_SP, mc.mc_gpregs.gp_sp);
  check_reg(&cursor, UNW_AARCH64_LR, mc.mc_gpregs.gp_lr);
  for (int i = 0; i < 30; ++i)
    check_reg(&cursor, UNW_AARCH64_X0 + i, mc.mc_gpregs.gp_x[i]);

  if (mc.mc_flags & _MC_FP_VALID) {
    for (int i = 0; i < 32; ++i) {
      unw_fpreg_t value;
      assert(unw_get_fpreg(&cursor, UNW_AARCH64_V0 + i, &value) == 0);
      assert(memcmp(&value, &mc.mc_fpregs.fp_q[i], sizeof(value)) == 0);
    }
  }

  assert(unw_step(&cursor) > 0);
  assert(!unw_is_signal_frame(&cursor));
  unw_word_t pc;
  assert(unw_get_reg(&cursor, UNW_REG_IP, &pc) == 0);
  assert(pc >= reinterpret_cast<uintptr_t>(FUNC_START(main_func)));
  assert(pc < reinterpret_cast<uintptr_t>(FUNC_END(main_func)));

  // Skip the BRK instruction when returning from the signal handler.
  uc->uc_mcontext.mc_gpregs.gp_elr += 4;
  handled = handled + 1;
}

__attribute__((noinline)) static void leaf() { asm volatile("brk #0"); }

FUNC_ATTR(main_func) int main(int, char **) {
  stack_t stack = {};
  stack.ss_size = SIGSTKSZ;
  stack.ss_sp = malloc(stack.ss_size);
  assert(stack.ss_sp != nullptr);
  assert(sigaltstack(&stack, nullptr) == 0);

  struct sigaction action = {};
  action.sa_sigaction = handler;
  action.sa_flags = SA_SIGINFO;
  assert(sigemptyset(&action.sa_mask) == 0);
  assert(sigaction(SIGTRAP, &action, nullptr) == 0);
  leaf();
  assert(handled == 1);

  action.sa_flags |= SA_ONSTACK;
  assert(sigaction(SIGTRAP, &action, nullptr) == 0);
  leaf();
  assert(handled == 2);

  stack.ss_flags = SS_DISABLE;
  assert(sigaltstack(&stack, nullptr) == 0);
  free(stack.ss_sp);
  return 0;
}
