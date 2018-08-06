// Test to check if we handle pic code properly.

// RUN: %clangxx_xray -fxray-instrument -std=c++11 -ffunction-sections \
// RUN:     -fdata-sections -fpic -fpie -Wl,--gc-sections %s -o %t
// RUN: rm -f pic-test-logging-*
// RUN: XRAY_OPTIONS="patch_premain=true verbosity=1 xray_naive_log=true \
// RUN:     xray_logfile_base=pic-test-logging-" %run %t 2>&1 | FileCheck %s
// After all that, clean up the output xray log.
//
// RUN: rm -f pic-test-logging-*

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include <cstdio>

[[clang::xray_always_instrument]]
unsigned short foo (unsigned b);

[[clang::xray_always_instrument]]
unsigned short bar (unsigned short a)
{
  printf("bar() is always instrumented!\n");
  return foo(a);
}

unsigned short foo (unsigned b)
{
  printf("foo() is always instrumented!\n");
  return b + b + 5;
}

int main ()
{
  // CHECK: XRay: Log file in 'pic-test-logging-{{.*}}'
  bar(10);
  // CHECK: bar() is always instrumented!
  // CHECK-NEXT: foo() is always instrumented!
}
