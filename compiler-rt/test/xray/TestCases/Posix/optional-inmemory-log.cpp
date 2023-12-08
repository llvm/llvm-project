// Make sure that we don't get the inmemory logging implementation enabled when
// we turn it off via options.

// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=true verbosity=1 xray_logfile_base=optional-inmemory-log.xray-" %run %t 2>&1 | FileCheck %s
//
// Make sure we clean out the logs in case there was a bug.
//
// RUN: rm -f optional-inmemory-log.xray-*

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include <cstdio>

[[clang::xray_always_instrument]] void foo() {
  printf("foo() is always instrumented!");
}

int main() {
  // CHECK-NOT: XRay: Log file in 'optional-inmemory-log.xray-{{.*}}'
  foo();
  // CHECK: foo() is always instrumented!
}
