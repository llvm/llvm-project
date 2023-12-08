// REQUIRES: static-analyzer
// RUN: clang-tidy %s -checks='-*,clang-analyzer-*' -- | FileCheck %s
extern void *malloc(unsigned long);
extern void free(void *);

void f() {
  int *p = new int(42);
  delete p;
  delete p;
  // CHECK: warning: Attempt to free released memory [clang-analyzer-cplusplus.NewDelete]
}

void g() {
  void *q = malloc(132);
  free(q);
  free(q);
  // CHECK: warning: Attempt to free released memory [clang-analyzer-unix.Malloc]
}
