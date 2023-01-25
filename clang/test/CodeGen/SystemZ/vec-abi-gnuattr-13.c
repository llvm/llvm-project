// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s \
// RUN:   -Wno-undefined-internal 2>&1 | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Passing vector argument to varargs function between static functions. This
// also potentially exposes the vector ABI as the va_list may be passed on to
// another (global) function.

typedef __attribute__((vector_size(16))) int v4i32;

static int bar(int N, ...);

static void foo() {
  v4i32 Var = {0, 0, 0, 0};
  bar(0, Var);
}

void fun() { foo(); }

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
