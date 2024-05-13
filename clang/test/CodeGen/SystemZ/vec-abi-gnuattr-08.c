// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Passing a struct containing a vector element.

typedef __attribute__((vector_size(16))) int v4i32;

struct S {
  int A;
  v4i32 B;
};

void bar(struct S Arg);

void foo() {
  struct S Var = {0, {0, 0, 0, 0}};
  bar(Var);
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
