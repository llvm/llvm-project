// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Using external vector variable (twice to test that the module flag is only
// added once, which would be an error).

typedef __attribute__((vector_size(16))) int v4i32;

extern v4i32 Var;

static void foo() {
  v4i32 Loc = {1, 1, 1, 1};
  Var = Var + Loc;
}

static void bar() {
  v4i32 Loc = {1, 2, 3, 4};
  Var = Var + Loc;
}

void fun1() { foo(); }
void fun2() { bar(); }

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
