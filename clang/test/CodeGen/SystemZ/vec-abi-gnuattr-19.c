// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Globally visible struct with function pointer array member with vector
// return values.

typedef __attribute__((vector_size(16))) int v4i32;

struct S {
  int i;
  v4i32 (*funcptr[4])(int);
};

struct S Arr[16];

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
