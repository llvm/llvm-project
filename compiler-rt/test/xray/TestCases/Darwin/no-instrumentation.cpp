// Verify that compiling without -fxray-instrument produces no xray sections.

// RUN: %clangxx %s -o %t
// RUN: otool -l %t | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// CHECK-NOT: xray_instr_map

#include <cstdio>

int main() {
  printf("no instrumentation\n");
  return 0;
}
