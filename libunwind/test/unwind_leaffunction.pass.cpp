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

  // Unwind until the main is reached, above frames depend on the platform and
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

__attribute__((noinline)) void crashing_leaf_func(int do_trap) {
  // libunwind searches for the address before the return address which points
  // to the trap instruction. We make the trap conditional and prevent inlining
  // of the function to ensure that the compiler doesn't remove the `ret`
  // instruction altogether.
  //
  // It's also important that the trap instruction isn't the first instruction
  // in the function (which it isn't because of the branch) for other unwinders
  // that also decrement pc.
  if (do_trap)
    __builtin_trap();
}

int main(int, char**) {
  signal(SIGTRAP, signal_handler);
  signal(SIGILL, signal_handler);
  crashing_leaf_func(1);
  return -2;
}
