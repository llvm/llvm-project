// RUN: %clangxx_asan %s -o %t -DSTACK_ALLOC_SIZE=8 -DREAD_SIZE=8
// RUN: not %run %t 1 2>&1 | FileCheck %s
// RUN: not %run %t 2 2>&1 | FileCheck %s
// RUN: not %run %t 7 2>&1 | FileCheck %s

// RUN: %clangxx_asan %s -o %t -DSTACK_ALLOC_SIZE=16 -DREAD_SIZE=16
// RUN: not %run %t 1 2>&1 | FileCheck %s
// RUN: not %run %t 2 2>&1 | FileCheck %s
// RUN: not %run %t 7 2>&1 | FileCheck %s
// RUN: not %run %t 8 2>&1 | FileCheck %s
// RUN: not %run %t 15 2>&1 | FileCheck %s

// RUN: %clangxx_asan %s -o %t -DSTACK_ALLOC_SIZE=16 -DREAD_SIZE=15
// RUN: not %run %t 2 2>&1 | FileCheck %s
// RUN: not %run %t 3 2>&1 | FileCheck %s
// RUN: not %run %t 7 2>&1 | FileCheck %s
// RUN: not %run %t 8 2>&1 | FileCheck %s
// RUN: not %run %t 15 2>&1 | FileCheck %s

// RUN: %clangxx_asan %s -o %t -DSTACK_ALLOC_SIZE=20 -DREAD_SIZE=18
// RUN: not %run %t 3 2>&1 | FileCheck %s
// RUN: not %run %t 7 2>&1 | FileCheck %s
// RUN: not %run %t 10 2>&1 | FileCheck %s
// RUN: not %run %t 13 2>&1 | FileCheck %s
// RUN: not %run %t 19 2>&1 | FileCheck %s

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

struct X {
  char bytes[READ_SIZE];
};

__attribute__((noinline)) struct X out_of_bounds(int offset) {
  volatile char bytes[STACK_ALLOC_SIZE];
  struct X* x_ptr = (struct X*)(bytes + offset);
  return *x_ptr;
}

int main(int argc, char **argv) {
  int offset = atoi(argv[1]);

  // Output READ_SIZE to check against the asan report,
  // and flush it so the output comes out first.
  fprintf(stderr, "Expected READ size: %d\n", READ_SIZE);
  fflush(stderr);

  // We are explicitly testing that we correctly detect and report this error
  // as a *partial stack buffer overflow*.
  assert(offset < STACK_ALLOC_SIZE);
  assert(offset + READ_SIZE > STACK_ALLOC_SIZE);

  struct X x = out_of_bounds(offset);
  int y = 0;

  for (int i = 0; i < READ_SIZE; i++) {
    y ^= x.bytes[i];
  }

  return y;
}

// CHECK: Expected READ size: [[READ_SIZE:[0-9]+]]
// CHECK: ERROR: AddressSanitizer: stack-buffer-overflow on address
// CHECK: READ of size [[READ_SIZE]] at {{0x.*}} thread T0
// CHECK: {{.* 0x.* in out_of_bounds.*stack-buffer-overflow-partial.cpp:}}
// CHECK: {{Address 0x.* is located in stack of thread T0 at offset}}
// CHECK: {{    #0 0x.* in out_of_bounds.*stack-buffer-overflow-partial.cpp:}}
// CHECK: 'bytes'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
