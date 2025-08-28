// REQUIRES: aarch64-registered-target || x86-registered-target

#include <ripple.h>

size_t check(int n) {
  ripple_block_t BS;
  if (n > 32)
    BS = ripple_set_block_shape(0, 32, 4);
  else
    BS = ripple_set_block_shape(0, 8);

  size_t rid = ripple_id(BS, 0);
  return ripple_reduceadd(0x1, rid);
}

// CHECK: setshape-invalid1.c:{{.*}}: block shape access is ambiguous (multiple shapes apply)
// CHECK: setshape-invalid1.c:{{.*}}: can come from here
// CHECK: setshape-invalid1.c:{{.*}}: and here

// RUN: %clang -fenable-ripple -O0 -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t
// RUN: %clang -fenable-ripple -O1 -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t
// RUN: %clang -fenable-ripple -O2 -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t
// RUN: %clang -fenable-ripple -O3 -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t
// RUN: %clang -fenable-ripple -Os -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t
// RUN: %clang -fenable-ripple -Oz -g -S -emit-llvm %s 2>%t; FileCheck %s --input-file=%t