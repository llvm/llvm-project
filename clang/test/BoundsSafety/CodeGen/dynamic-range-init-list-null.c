

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null 2> /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null 2> /dev/null
// This is to ensure no crash in code gen. A general version of filecheck is in dynamic-range-init-list.c.

#include <ptrcheck.h>

struct RangePtrs {
  int *__ended_by(iter) start;
  int *__ended_by(end) iter;
  void *end;
};

void Test1(void) {
  int arr[10];
  int *ptr = arr;
  struct RangePtrs rptrs = { ptr, 0, arr + 10 };
}

void Test2(void) {
  int arr[10];
  int *ptr = arr;
  struct RangePtrs rptrs = { ptr, arr, 0 };
}

void Test3(void) {
  int arr[10];
  int *ptr = arr;
  struct RangePtrs rptrs = { 0, arr, arr + 1 };
}
