// RUN: %clang_asan -O0 %s -o %t
// RUN: %env_asan_opts=verbosity=2 %run %t 2>&1 | FileCheck %s
// REQUIRES: powerpc64-target-arch

#include <stdio.h>

int main() {
  // CHECK: || `[0xfffff8000000000, 0xfffffffffffffff]` || HighMem    ||
  // CHECK: || `[0xa80fff000000000, 0xa80ffffffffffff]` || HighShadow ||
  // CHECK: || `[0xa00000000000000, 0xa0001ffffffffff]` || MidMem     ||
  // CHECK: || `[0xa41000000000000, 0xa41003fffffffff]` || MidShadow  ||
  // CHECK: || `[0xa21020000000000, 0xa21020fffffffff]` || Mid2Shadow  ||
  // CHECK: || `[0xa01020000000000, 0xa01020fffffffff]` || Mid3Shadow  ||
  // CHECK: || `[0xa01000000000000, 0xa01000fffffffff]` || LowShadow  ||
  // CHECK: || `[0x000000000000, 0x007fffffffff]` || LowMem     ||

  return 0;
}
