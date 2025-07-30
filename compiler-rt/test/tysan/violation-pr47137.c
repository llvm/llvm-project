// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// https://github.com/llvm/llvm-project/issues/47137
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void f(int m) {
  int n = (4 * m + 2) / 3;
  uint64_t *a = malloc(n * sizeof(uint64_t));
  uint64_t *b = malloc(n * sizeof(uint64_t));
  uint64_t aa[] = {0xffff3e0000000001, 0x22eaf0b680a88c16, 0x5a65d25ac40e20f3,
                   0x34e7ac346236953e, 0x9dea3e0a26c6ba89, 0x0000000000000000,
                   0x0000000000000000, 0x0000000000000000};
  uint64_t bb[] = {0x0000000024c0ffff, 0x000000004634d940, 0x00000000219d18ef,
                   0x0000000000154519, 0x000000000000035f, 0x0000000000000000,
                   0x0000000000000000, 0x0000000000000000};
  char l[20];
  l[0] = 0;
  for (int i = 0; i < n; i++) {
    a[i] = aa[i] + l[0] - '0';
    b[i] = bb[i] + l[0] - '0';
  }

  // CHECK:      TypeSanitizer: type-aliasing-violation on address
  // CHECK-NEXT: READ of size 2 at {{.+}} with type short accesses an existing object of type long
  // CHECK-NEXT:    in f {{.*/?}}violation-pr47137.c:31
  for (int i = 0, j = 0; j < 4 * m; i += 4, j += 3) {
    for (int k = 0; k < 3; k++) {
      ((uint16_t *)a)[j + k] = ((uint16_t *)a)[i + k];
      ((uint16_t *)b)[j + k] = ((uint16_t *)b)[i + k];
    }
  }

  printf("a: %016llx\n", a[0]);
  free(a);
  free(b);
}

int main() { f(6); }
