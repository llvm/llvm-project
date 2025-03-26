// Check that __asan_poison_memory_region and ASAN_OPTIONS=track_poison work.
//
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=track_poison=1000 not %run %t       2>&1 | FileCheck %s --check-prefixes=CHECK-AC,CHECK-A
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=track_poison=1000     %run %t 20    2>&1 | FileCheck %s --check-prefixes=CHECK-B
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=track_poison=1000 not %run %t 30 30 2>&1 | FileCheck %s --check-prefixes=CHECK-AC,CHECK-C

#include <stdio.h>
#include <stdlib.h>

extern "C" void __asan_poison_memory_region(void *, size_t);
extern "C" void __asan_unpoison_memory_region(void *, size_t);

void novichok(char *x) {
  __asan_poison_memory_region(x, 64);       // A
  __asan_unpoison_memory_region(x + 16, 8); // B
  __asan_poison_memory_region(x + 24, 16);  // C
}

void fsb(char *x) { novichok(x); }

int main(int argc, char **argv) {
  char *x = new char[64];
  x[10] = 0;
  fsb(x);
  // Bytes [ 0, 15]: poisoned by A
  // Bytes [16, 23]: unpoisoned by B
  // Bytes [24, 63]: poisoned by C

  int res = x[argc * 10]; // BOOOM
  // CHECK-AC: ERROR: AddressSanitizer: use-after-poison
  // CHECK-AC: main{{.*}}use-after-poison-tracked.cpp:[[@LINE-2]]
  // CHECK-B-NOT: ERROR: AddressSanitizer: use-after-poison

  // CHECK-AC: Memory was manually poisoned by thread T0:
  // CHECK-A: novichok{{.*}}use-after-poison-tracked.cpp:[[@LINE-21]]
  // CHECK-C: novichok{{.*}}use-after-poison-tracked.cpp:[[@LINE-20]]
  // CHECK-AC: fsb{{.*}}use-after-poison-tracked.cpp:[[@LINE-18]]
  // CHECK-AC: main{{.*}}use-after-poison-tracked.cpp:[[@LINE-14]]
  // CHECK-B-NOT: Memory was manually poisoned by thread T0:

  delete[] x;

  printf("End of program reached\n");
  // CHECK-B: End of program reached

  return 0;
}
