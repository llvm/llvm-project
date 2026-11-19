// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux
// UNSUPPORTED: libunwind-arm-ehabi

// Inline assembly isn't supported by Memory Sanitizer
// UNSUPPORTED: msan

// Unwinding through a frame interprets its CFI up to the PC. Every
// DW_CFA_remember_state that is later matched by a DW_CFA_restore_state must
// not keep its saved state alive: a function with many epilogues in the middle
// of its code has one such pair per epilogue, and the unwinder must not need
// stack in proportion to their number. Here the pairs would take several MiB,
// far more than the stack is allowed to grow to.

#undef NDEBUG
#include <assert.h>
#include <sys/resource.h>
#include <unwind.h>

static _Unwind_Reason_Code count_frames(struct _Unwind_Context *, void *arg) {
  ++*static_cast<int *>(arg);
  return _URC_NO_REASON;
}

__attribute__((noinline)) static int unwind_through_remember_states() {
  // Emits no instructions, only 5000 remember/restore pairs in this function's
  // FDE. They precede the call, so unwinding from it interprets all of them.
  asm volatile(".rept 5000\n.cfi_remember_state\n.cfi_restore_state\n.endr");
  int frames = 0;
  _Unwind_Backtrace(count_frames, &frames);
  return frames;
}

int main(int, char **) {
  // The main thread's stack only grows up to the soft limit.
  struct rlimit limit;
  assert(getrlimit(RLIMIT_STACK, &limit) == 0);
  limit.rlim_cur = 256 * 1024;
  assert(setrlimit(RLIMIT_STACK, &limit) == 0);

  // At least `unwind_through_remember_states` and `main`.
  assert(unwind_through_remember_states() >= 2);
  return 0;
}
