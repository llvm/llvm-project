// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that leaf function can be unwund.
// REQUIRES: x86_64-linux

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
  assert(dladdr((void*)_Unwind_GetIP(ctx), &info));

  // Unwind util the main is reached, above frames deeped on the platfrom and architecture.
  if(info.dli_sname && !strcmp("main", info.dli_sname)) {
    _Exit(0);
  }
  return _URC_NO_REASON;
}

void signal_handler(int signum) {
  (void)signum;
  _Unwind_Backtrace(frame_handler, NULL);
  _Exit(-1);
}

int* faultyPointer = NULL;

__attribute__((noinline)) void crashing_leaf_func(void) {
  *faultyPointer = 0;
}

int main(int, char**) {
  signal(SIGSEGV, signal_handler);
  crashing_leaf_func();
  return -2;
}