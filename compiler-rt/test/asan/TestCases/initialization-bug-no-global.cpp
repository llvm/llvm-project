// RUN: %clangxx_asan %min_macos_deployment_target=10.11 -O0 %s %p/Helpers/initialization-bug-extra.cpp -o %t
// RUN: %env_asan_opts=check_initialization_order=true:strict_init_order=true not %run %t 2>&1 | FileCheck %s

// Do not test with optimization -- the error may be optimized away.

// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=186
// XFAIL: target={{.*windows-msvc.*}}

// Fails on some Darwin bots, probably iOS.
// XFAIL: ios

#include <stdio.h>

extern int y;

void __attribute__((constructor)) ctor() {
  printf("%d\n", y);
  // CHECK: AddressSanitizer: initialization-order-fiasco
}

int main() {
  // ASan should have caused an exit before main runs.
  printf("PASS\n");
  // CHECK-NOT: PASS
  return 0;
}
