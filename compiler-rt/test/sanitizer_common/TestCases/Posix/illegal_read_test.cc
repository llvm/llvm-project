// Test that there was an illegal READ memory access.
// RUN: %clangxx -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime
// XFAIL: powerpc64

volatile int *null = 0;
volatile int a;

int main(int argc, char **argv) {
  a = *null;
  return 0;
}

// CHECK: The signal is caused by a READ memory access.
