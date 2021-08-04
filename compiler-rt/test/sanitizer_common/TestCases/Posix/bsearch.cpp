// Check interceptor.
// RUN: %clangxx -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// Inlined bsearch works even without interceptors.
// RUN: %clangxx -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

static int arr1[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
static int arr2[] = {10, 1, 1, 3, 4, 6, 7, 7};

#define array_size(x) (sizeof(x) / sizeof(x[0]))

static int cmp_ints(const void *a, const void *b) {
  return *(const int *)b - *(const int *)a;
}

static int cmp_pos(const void *a, const void *b) {
  const int *ap =
      (const int *)bsearch(a, arr1, array_size(arr1), sizeof(int), &cmp_ints);
  if (!ap)
    ap = arr1 + array_size(arr1);
  const int *bp =
      (const int *)bsearch(b, arr1, array_size(arr1), sizeof(int), &cmp_ints);
  if (!bp)
    bp = arr1 + array_size(arr1);
  return bp - ap;
}

int main() {
  // Simple bsearch.
  for (int i = 0; i < 10; ++i) {
    const void *r =
        bsearch(&i, arr1, array_size(arr1), sizeof(arr1[0]), &cmp_ints);
    if (!r)
      printf(" null");
    else
      printf(" %d", *(const int *)r);
  }
  printf("\n");
  // CHECK: 0 1 2 3 4 5 6 7 8 null

  // Nested bsearch.
  for (int i = 0; i < 10; ++i) {
    const void *r =
        bsearch(&i, arr2, array_size(arr2), sizeof(arr2[0]), &cmp_pos);
    if (!r)
      printf(" null");
    else
      printf(" %d", *(const int *)r);
  }
  printf("\n");
  // CHECK: null 1 null 3 4 null 6 7 null 10
}
