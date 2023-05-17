// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Defining globally visible function with vector return value.

typedef __attribute__((vector_size(16))) int v4i32;

v4i32 fun(int v) {
  v4i32 Val = {v, v, v, v};
  return Val;
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
