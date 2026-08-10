// RUN: %clangxx_asan -O0 -std=c++11 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 -std=c++11 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 -std=c++11 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test that we do not detect false buffer overflows cased by optimization when
// when local variable replaced by a smaller global constant.
// https://bugs.llvm.org/show_bug.cgi?id=33372

#include <stdio.h>
#include <string.h>

struct A { int x, y, z; };
struct B { A a; /*gap*/ long b; };
B *bb;

void test1() {
  A a1 = {1, 1, 2};
  B b1 = {a1, 6};
  bb = new B(b1);
}

const char KKK[] = {1, 1, 2};
char bbb[100000];

void test2() {
  char cc[sizeof(bbb)];
  memcpy(cc, KKK , sizeof(KKK));
  memcpy(bbb, cc, sizeof(bbb));
}

int main(int argc, char *argv[]) {
  test1();
  test2();
  printf("PASSED");
  return 0;
}

// CHECK-NOT: ERROR: AddressSanitizer
// CHECK: PASSED
