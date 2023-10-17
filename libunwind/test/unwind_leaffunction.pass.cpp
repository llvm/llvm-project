// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that leaf function can be unwund.
// REQUIRES: target={{(aarch64|riscv64|s390x|x86_64)-.+linux.*}}

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

#undef NDEBUG
#include <assert.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

_Unwind_Reason_Code frame_handler(struct _Unwind_Context* ctx, void* arg) {
  (void)arg;
  Dl_info info = { 0, 0, 0, 0 };

  // Unwind until the main is reached, above frames deeped on the platform and
  // architecture.
  if (dladdr(reinterpret_cast<void *>(_Unwind_GetIP(ctx)), &info) &&
      info.dli_sname && !strcmp("main", info.dli_sname)) {
    _Exit(0);
  }
  return _URC_NO_REASON;
}

void signal_handler(int signum) {
  (void)signum;
  _Unwind_Backtrace(frame_handler, NULL);
  _Exit(-1);
}

__attribute__((noinline)) void crashing_leaf_func(void) {
  // libunwind searches for the address before the return address which points
  // to the trap instruction. NOP guarantees the trap instruction is not the
  // first instruction of the function.
  // We should keep this here for other unwinders that also decrement pc.
  __asm__ __volatile__("nop");
  __builtin_trap();
}

int main(int, char**) {
  signal(SIGTRAP, signal_handler);
  signal(SIGILL, signal_handler);
  crashing_leaf_func();
  return -2;
}
