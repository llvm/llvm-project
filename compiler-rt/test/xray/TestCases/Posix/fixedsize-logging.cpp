// Check to make sure that we have a log file with a fixed-size.

// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic verbosity=1 xray_logfile_base=fixedsize-logging-" %run %t 2>&1 | FileCheck %s
//
// After all that, clean up the output xray log.
//
// RUN: rm fixedsize-logging-*

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include <cstdio>

[[clang::xray_always_instrument]] void foo() {
  printf("foo() is always instrumented!");
}

int main() {
  // CHECK: XRay: Log file in 'fixedsize-logging-{{.*}}'
  foo();
  // CHECK: foo() is always instrumented!
}
