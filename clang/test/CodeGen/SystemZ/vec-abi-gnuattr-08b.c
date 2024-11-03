// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Passing a single element struct containing a narrow (8 byte) vector element.

typedef __attribute__((vector_size(8))) int v2i32;

struct S {
  v2i32 B;
};

void bar(struct S Arg);

void foo() {
  struct S Var = {{0, 0}};
  bar(Var);
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
