// Test that __orc_rt_coff_jit_dlopen and __orc_rt_coff_jit_dlclose work as
// expected for a straightforward dlopen; dlclose sequence: first the
// constructors should be run.
//
// RUN: %clang_cl -MD -c -o %t.inits.o %p/Inputs/standalone-dylib.c
// RUN: %clang_cl -MD -c -o %t.test.o %s
// RUN: %llvm_jitlink \
// RUN:   -alias Platform:dlopen=__orc_rt_coff_jit_dlopen \
// RUN:   -alias Platform:dlclose=__orc_rt_coff_jit_dlclose \
// RUN:   %t.test.o -jd inits %t.inits.o -lmain | FileCheck %s

// CHECK: entering main
// CHECK-NEXT: constructor
// CHECK-NEXT: destructor
// CHECK-NEXT: leaving main

#include <stdio.h>
void *dlopen(const char *path, int mode);
int dlclose(void *handle);

int main(int argc, char *argv[]) {
  printf("entering main\n");
  void *H = dlopen("inits", 0);
  if (!H) {
    printf("failed\n");
    return -1;
  }
  if (dlclose(H) == -1) {
    printf("failed\n");
    return -1;
  }
  printf("leaving main\n");
  return 0;
}
