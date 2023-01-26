// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Passing address of local function with vector arg to global function.

typedef __attribute__((vector_size(16))) int v4i32;

void GlobFun(v4i32 (*f)(v4i32));

static v4i32 foo(v4i32 Arg) {
  return Arg;
}

void fun() {
  GlobFun(foo);
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
