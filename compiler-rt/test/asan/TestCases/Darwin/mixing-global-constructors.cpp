// A global constructor from a non-instrumented part calls a function
// in an instrumented part.
// Regression test for https://code.google.com/p/address-sanitizer/issues/detail?id=363.

// RUN: %clangxx      -DINSTRUMENTED_PART=0 -c %s -o %t-uninst.o
// RUN: %clangxx_asan -DINSTRUMENTED_PART=1 -c %s -o %t-inst.o
// RUN: %clangxx_asan %t-uninst.o %t-inst.o -o %t

// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void func(char *ptr);

#if INSTRUMENTED_PART == 1

void func(char *ptr) {
  *ptr = 'X';
}

#else // INSTRUMENTED_PART == 1

struct C1 {
  C1() {
    printf("Hello ");
    char buffer[10] = "world";
    func(buffer);
    printf("%s\n", buffer);
  }
};

C1 *obj = new C1();

int main(int argc, const char *argv[]) {
  return 0;
}

#endif // INSTRUMENTED_PART == 1

// CHECK: Hello Xorld
