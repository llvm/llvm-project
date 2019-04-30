// RUN: %clang_cl_asan %s -o %t.exe
// RUN: %run %t.exe 2>&1 | FileCheck %s
// RUN: %clang_cl %s -o %t.exe
// RUN: %run %t.exe 2>&1 | FileCheck %s

#include <cassert>
#include <stdio.h>
#include <windows.h>

int main() {
  void *p = calloc(1, 100);
  assert(p);
  void *np = _recalloc(p, 2, 100);
  assert(np);
  for (int i = 0; i < 2 * 100; i++) {
    assert(((BYTE *)np)[i] == 0);
  }
  void *nnp = _recalloc(np, 1, 100);
  assert(nnp);
  for (int i = 0; i < 100; i++) {
    assert(((BYTE *)nnp)[i] == 0);
    ((BYTE *)nnp)[i] = 0x0d;
  }
  void *nnnp = _recalloc(nnp, 2, 100);
  assert(nnnp);
  for (int i = 0; i < 100; i++) {
    assert(((BYTE *)nnnp)[i] == 0x0d);
  }
  for (int i = 100; i < 200; i++) {
    assert(((BYTE *)nnnp)[i] == 0);
  }
  fprintf(stderr, "passed\n");
  return 0;
}

// CHECK-NOT: Assertion
// CHECK: passed