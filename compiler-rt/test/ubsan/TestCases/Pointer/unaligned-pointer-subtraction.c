// Check that -fsanitize=unaligned-pointer-subtraction reports, at runtime, a
// subtraction of two pointers whose byte distance is not an exact multiple of
// the element size (i.e. the operands do not point into the same array).
//
// RUN: %clang -fsanitize=unaligned-pointer-subtraction %s -o %t
// RUN: %run %t a 2>&1 | FileCheck %s --check-prefix=CONST
// RUN: %run %t n 2>&1 | FileCheck %s --check-prefix=NEG
// RUN: %run %t v 2>&1 | FileCheck %s --check-prefix=VLA
// RUN: %run %t o 2>&1 | FileCheck %s --check-prefix=SAFE --implicit-check-not="runtime error:"
//
// Fatal (non-recoverable) mode aborts after the first report.
// RUN: %clang -fsanitize=unaligned-pointer-subtraction \
// RUN:     -fno-sanitize-recover=unaligned-pointer-subtraction %s -o %t.abort
// RUN: not %run %t.abort a 2>&1 | FileCheck %s --check-prefix=ABORT

#include <stdio.h>

typedef struct {
  int x, y;
} A; // sizeof(A) == 8

// Constant element size: byte distance 4 is not a multiple of 8.
__attribute__((noinline)) long sub_const(int *p) {
  return (A *)(p + 1) - (A *)p;
}

// Negative distance: exercises signed formatting of the reported distance.
__attribute__((noinline)) long sub_negative(int *p) {
  return (A *)p - (A *)(p + 1);
}

// VLA element type: the element size (sizeof(int) * n) is a runtime value.
__attribute__((noinline)) long sub_vla(int n, int (*p)[n]) {
  int(*q)[n] = (int(*)[n])((char *)p + 4);
  return q - p;
}

// Valid subtraction: byte distance 8 is a multiple of 8, so no report.
__attribute__((noinline)) long sub_ok(int *p) { return (A *)(p + 2) - (A *)p; }

// CONST: runtime error: pointer subtraction with byte distance 4 that is not a multiple of the element size 8
// NEG: runtime error: pointer subtraction with byte distance -4 that is not a multiple of the element size 8
// VLA: runtime error: pointer subtraction with byte distance 4 that is not a multiple of the element size 12
// ABORT: runtime error: pointer subtraction with byte distance 4 that is not a multiple of the element size 8
// SAFE: r=1
int main(int argc, char **argv) {
  static int a[8];
  char c = argc > 1 ? argv[1][0] : 'a';
  long r = 0;
  switch (c) {
  case 'a':
    r = sub_const(a);
    break;
  case 'n':
    r = sub_negative(a);
    break;
  case 'v':
    r = sub_vla(3, (int(*)[3])a);
    break;
  case 'o':
    r = sub_ok(a);
    break;
  }
  printf("r=%ld\n", r);
  return 0;
}
