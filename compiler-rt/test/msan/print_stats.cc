// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -g %s -o %t 
// RUN: %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-NOSTATS %s
// RUN: MSAN_OPTIONS=print_stats=1 %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-NOSTATS %s
// RUN: MSAN_OPTIONS=print_stats=1,atexit=1 %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-STATS %s

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -g -DPOSITIVE=1 %s -o %t 
// RUN: not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-NOSTATS %s
// RUN: MSAN_OPTIONS=print_stats=1 not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-STATS %s

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -fsanitize-recover=memory -g -DPOSITIVE=1 %s -o %t
// RUN: not %run %t 2>&1 | \
// RUN:  FileCheck --check-prefixes=CHECK,CHECK-NOSTATS,CHECK-RECOVER %s
// RUN: MSAN_OPTIONS=print_stats=1 not %run %t 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK-STATS,CHECK-RECOVER %s

#include <stdio.h>
int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  fprintf(stderr, "TEST\n");
#ifdef POSITIVE
  return *p;
#else
  return 0;
#endif
}

// CHECK: TEST

// CHECK-STATS: Unique heap origins:
// CHECK-STATS: Stack depot allocated bytes:
// CHECK-STATS: Unique origin histories:
// CHECK-STATS: History depot allocated bytes:

// CHECK-NOSTATS-NOT: Unique heap origins:
// CHECK-NOSTATS-NOT: Stack depot allocated bytes:
// CHECK-NOSTATS-NOT: Unique origin histories:
// CHECK-NOSTATS-NOT: History depot allocated bytes:

// CHECK-RECOVER: MemorySanitizer: 1 warnings reported.
