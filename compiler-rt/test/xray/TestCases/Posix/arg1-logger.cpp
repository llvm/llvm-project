// Check that we can get the first function argument logged
// using a custom logging function.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: rm -f arg1-logger-*
// RUN: env XRAY_OPTIONS="patch_premain=true verbosity=1 xray_mode=xray-basic \
// RUN:    xray_logfile_base=arg1-logger-" %run %t 2>&1 | FileCheck %s
//
// After all that, clean up the XRay log file.
//
// RUN: rm -f arg1-logger-*

// REQUIRES: target={{(aarch64|x86_64)-.*}}

#include "xray/xray_interface.h"

#include <cinttypes>
#include <cstdio>

void arg1logger(int32_t fn, XRayEntryType t, uint64_t a1) {
  printf("Arg1: %" PRIx64 ", XRayEntryType %u\n", a1, t);
}

[[clang::xray_always_instrument, clang::xray_log_args(1)]] void foo(void *) {}

int main() {
  // CHECK: XRay: Log file in 'arg1-logger-{{.*}}'

  __xray_set_handler_arg1(arg1logger);
  foo(nullptr);
  // CHECK: Arg1: 0, XRayEntryType 3

  __xray_remove_handler_arg1();
  foo((void *) 0xBADC0DE);
  // nothing expected to see here

  __xray_set_handler_arg1(arg1logger);
  foo((void *) 0xDEADBEEFCAFE);
  // CHECK-NEXT: Arg1: deadbeefcafe, XRayEntryType 3
  foo((void *) -1);
  // CHECK-NEXT: Arg1: ffffffffffffffff, XRayEntryType 3
}
