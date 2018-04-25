// Test that there was an illegal WRITE memory access.
// RUN: %clangxx -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: stable-runtime
// XFAIL: powerpc64

volatile int *null = 0;

int main(int argc, char **argv) {
  *null = 0;
  return 0;
}

// CHECK: The signal is caused by a WRITE memory access.
