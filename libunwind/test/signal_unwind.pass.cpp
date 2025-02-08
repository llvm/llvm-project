// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that the unwinder can cope with the signal handler.
// REQUIRES: target={{(aarch64|riscv64|s390x|x86_64)-.+linux.*}}

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

// Note: this test fails on musl because:
//
//  (a) musl disables emission of unwind information for its build, and
//  (b) musl's signal trampolines don't include unwind information
//
// XFAIL: target={{.*}}-musl

#undef NDEBUG
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

// Using __attribute__((section("main_func"))) is ELF specific, but then
// this entire test is marked as requiring Linux, so we should be good.
//
// We don't use dladdr() because on musl it's a no-op when statically linked.
extern char __start_main_func;
extern char __stop_main_func;

_Unwind_Reason_Code frame_handler(struct _Unwind_Context* ctx, void* arg) {
  (void)arg;

  // Unwind until the main is reached, above frames depend on the platform and
  // architecture.
  uintptr_t ip = _Unwind_GetIP(ctx);
  if (ip >= (uintptr_t)&__start_main_func &&
      ip < (uintptr_t)&__stop_main_func) {
    _Exit(0);
  }

  return _URC_NO_REASON;
}

void signal_handler(int signum) {
  (void)signum;
  _Unwind_Backtrace(frame_handler, NULL);
  _Exit(-1);
}

__attribute__((section("main_func"))) int main(int, char **) {
  signal(SIGUSR1, signal_handler);
  kill(getpid(), SIGUSR1);
  return -2;
}
