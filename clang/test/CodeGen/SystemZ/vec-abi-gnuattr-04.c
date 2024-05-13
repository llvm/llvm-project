// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Call via global function pointer in internal function, with vector argument.

typedef __attribute__((vector_size(16))) int v4i32;

void (*bar)(v4i32 Arg);

static void foo() {
  v4i32 Var = {0, 0, 0, 0};
  (*bar)(Var);
}

void fun() { foo(); }

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
