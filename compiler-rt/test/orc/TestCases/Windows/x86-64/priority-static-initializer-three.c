// RUN: %clang_cl -MD -c -o %t %s
// RUN: %llvm_jitlink %t 2>&1 | FileCheck %s
// CHECK: init1
// CHECK-NEXT: init2
// CHECK-NEXT: init3

#include <stdio.h>

int init1() {
  printf("init1\n");
  return 0;
}

int init2() {
  printf("init2\n");
  return 0;
}

int init3() {
  printf("init3\n");
  return 0;
}

#pragma section(".CRT$XIX", long, read)
__declspec(allocate(".CRT$XIX")) int (*i3)(void) = init3;

#pragma section(".CRT$XIV", long, read)
__declspec(allocate(".CRT$XIV")) int (*i1)(void) = init1;

#pragma section(".CRT$XIW", long, read)
__declspec(allocate(".CRT$XIW")) int (*i2)(void) = init2;

int main(int argc, char *argv[]) {
  fflush(stdout);
  return 0;
}
