// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Call via global function pointer in internal function, with vector return
// value.

typedef __attribute__((vector_size(16))) int v4i32;

v4i32 (*bar)(int);

static int foo() {
  return (*bar)(0)[0];
}

int fun() { return foo(); }

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
