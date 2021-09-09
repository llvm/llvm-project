// RUN: %clang -fno-use-init-array -g -c %s -o %t.o
// RUN: %clang -fno-use-init-array -g -o %t -nostdlib %crt1 %crti %crtbegin %t.o -lc %libgcc %crtend %crtn
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

// Ensure the various startup functions are called in the proper order.

// CHECK: __register_frame_info()
/// ctor() is here if ld.so/libc supports DT_INIT/DT_FINI
// CHECK:      main()
/// dtor() is here if ld.so/libc supports DT_INIT/DT_FINI
// CHECK:      __deregister_frame_info()

struct object;
static int counter;

void __register_frame_info(const void *fi, struct object *obj) {
  printf("__register_frame_info()\n");
}

void __deregister_frame_info(const void *fi) {
  printf("__deregister_frame_info()\n");
}

void __attribute__((constructor)) ctor() {
  printf("ctor()\n");
  ++counter;
}

void __attribute__((destructor)) dtor() {
  printf("dtor()\n");
  if (--counter != 0)
    abort();
}

int main() {
  printf("main()\n");
  return 0;
}
