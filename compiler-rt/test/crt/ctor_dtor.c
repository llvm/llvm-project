// RUN: %clang -fno-use-init-array -g -c %s -o %t.o
// RUN: %clang -fno-use-init-array -g -o %t -nostdlib %crt1 %crti %crtbegin %t.o -lc %libgcc %crtend %crtn
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>

// Ensure the various startup functions are called in the proper order.

// CHECK: __register_frame_info()
// CHECK-NEXT: ctor()
// CHECK-NEXT: main()
// CHECK-NEXT: dtor()
// CHECK-NEXT: __deregister_frame_info()

struct object;

void __register_frame_info(const void *fi, struct object *obj) {
  printf("__register_frame_info()\n");
}

void __deregister_frame_info(const void *fi) {
  printf("__deregister_frame_info()\n");
}

void __attribute__((constructor)) ctor() {
  printf("ctor()\n");
}

void __attribute__((destructor)) dtor() {
  printf("dtor()\n");
}

int main() {
  printf("main()\n");
  return 0;
}
