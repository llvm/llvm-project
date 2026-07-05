// Check that a qsort() comparator that calls qsort() works as expected
// RUN: %clangxx -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

struct Foo {
  int array[2];
};
int global_array[12] = {7, 11, 9, 10, 1, 2, 4, 3, 6, 5, 8, 12};

#define array_size(x) (sizeof(x) / sizeof(x[0]))

int ascending_compare_ints(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

int descending_compare_ints(const void *a, const void *b) {
  // Add another qsort() call to check more than one level of recursion
  qsort(global_array, array_size(global_array), sizeof(int), &ascending_compare_ints);
  return *(const int *)b - *(const int *)a;
}

int sort_and_compare(const void *a, const void *b) {
  struct Foo *f1 = (struct Foo *)a;
  struct Foo *f2 = (struct Foo *)b;
  printf("sort_and_compare({%d, %d}, {%d, %d})\n", f1->array[0], f1->array[1],
         f2->array[0], f2->array[1]);
  // Call qsort from within qsort() to check that interceptors handle this case:
  qsort(&f1->array, array_size(f1->array), sizeof(int), &descending_compare_ints);
  qsort(&f2->array, array_size(f2->array), sizeof(int), &descending_compare_ints);
  // Sort by second array element:
  return f1->array[1] - f2->array[1];
}

int main() {
  // Note: 16 elements should be large enough to trigger a recursive qsort() call.
  struct Foo qsortArg[16] = {
      {1, 99},
      {2, 3},
      {17, 5},
      {8, 6},
      {11, 4},
      {3, 3},
      {16, 17},
      {7, 9},
      {21, 12},
      {32, 23},
      {13, 8},
      {99, 98},
      {41, 42},
      {42, 43},
      {44, 45},
      {0, 1},
  };
  // Sort the individual arrays in descending order and the over all struct
  // Foo array in ascending order of the second array element.
  qsort(qsortArg, array_size(qsortArg), sizeof(qsortArg[0]), &sort_and_compare);

  printf("Sorted result:");
  for (const auto &f : qsortArg) {
    printf(" {%d,%d}", f.array[0], f.array[1]);
  }
  printf("\n");
  // CHECK: Sorted result: {1,0} {99,1} {3,2} {3,3} {11,4} {17,5} {8,6} {9,7} {13,8} {21,12} {17,16} {32,23} {42,41} {43,42} {45,44} {99,98}
  printf("Sorted global_array:");
  for (int i : global_array) {
    printf(" %d", i);
  }
  printf("\n");
  // CHECK: Sorted global_array: 1 2 3 4 5 6 7 8 9 10 11 12
}
