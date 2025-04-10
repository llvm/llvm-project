// Check that __asan_poison_memory_region and ASAN_OPTIONS=poison_history_size work.
//
// Poisoned access with history
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t       2>&1 | FileCheck %s --check-prefixes=CHECK-ACDE,CHECK-ABC,CHECK-AC,CHECK-A
//
// Not poisoned access
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000     %run %t 20    2>&1 | FileCheck %s --check-prefixes=CHECK-ABC,CHECK-B,CHECK-BDE
//
// Poisoned access with history (different stack trace)
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1000 not %run %t 30 30 2>&1 | FileCheck %s --check-prefixes=CHECK-ACDE,CHECK-ABC,CHECK-AC,CHECK-C
//
// Poisoned access without history
// RUN: %clangxx_asan -O0 %s -o %t &&                                           not %run %t       2>&1 | FileCheck %s --check-prefixes=CHECK-ACDE,CHECK-BDE,CHECK-D
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=0    not %run %t       2>&1 | FileCheck %s --check-prefixes=CHECK-ACDE,CHECK-BDE,CHECK-D

// Poisoned access with insufficient history
// RUN: %clangxx_asan -O0 %s -o %t && env ASAN_OPTIONS=poison_history_size=1    not %run %t       2>&1 | FileCheck %s --check-prefixes=CHECK-ACDE,CHECK-BDE,CHECK-E

#include <stdio.h>
#include <stdlib.h>

extern "C" void __asan_poison_memory_region(void *, size_t);
extern "C" void __asan_unpoison_memory_region(void *, size_t);

void honey_ive_poisoned_the_memory(char *x) {
  __asan_poison_memory_region(x, 64);       // A
  __asan_unpoison_memory_region(x + 16, 8); // B
  __asan_poison_memory_region(x + 24, 16);  // C
}

void foo(char *x) { honey_ive_poisoned_the_memory(x); }

int main(int argc, char **argv) {
  char *x = new char[64];
  x[10] = 0;
  foo(x);
  // Bytes [ 0, 15]: poisoned by A
  // Bytes [16, 23]: unpoisoned by B
  // Bytes [24, 63]: poisoned by C

  int res = x[argc * 10]; // BOOOM
  // CHECK-ACDE: ERROR: AddressSanitizer: use-after-poison
  // CHECK-ACDE: main{{.*}}use-after-poison-history-size.cpp:[[@LINE-2]]
  // CHECK-B-NOT: ERROR: AddressSanitizer: use-after-poison
  // CHECK-ABC-NOT: try the experimental setting ASAN_OPTIONS=poison_history_size=
  // CHECK-D: try the experimental setting ASAN_OPTIONS=poison_history_size=

  // CHECK-AC: Memory was manually poisoned by thread T0:
  // CHECK-A: honey_ive_poisoned_the_memory{{.*}}use-after-poison-history-size.cpp:[[@LINE-23]]
  // CHECK-C: honey_ive_poisoned_the_memory{{.*}}use-after-poison-history-size.cpp:[[@LINE-22]]
  // CHECK-AC: foo{{.*}}use-after-poison-history-size.cpp:[[@LINE-20]]
  // CHECK-AC: main{{.*}}use-after-poison-history-size.cpp:[[@LINE-16]]
  // CHECK-BDE-NOT: Memory was manually poisoned by thread T0:

  // CHECK-ABC-NOT: Try a larger value for ASAN_OPTIONS=poison_history_size=
  // CHECK-D-NOT: Try a larger value for ASAN_OPTIONS=poison_history_size=
  // CHECK-E: Try a larger value for ASAN_OPTIONS=poison_history_size=

  delete[] x;

  printf("End of program reached\n");
  // CHECK-B: End of program reached

  return 0;
}
