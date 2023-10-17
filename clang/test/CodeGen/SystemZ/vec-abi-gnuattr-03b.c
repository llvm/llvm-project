// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test that the "s390x-visible-vector-ABI" module flag is emitted.

// Call to external function with with narrow vector argument.

typedef __attribute__((vector_size(8))) int v2i32;

void bar(v2i32 arg);

void foo() {
  v2i32 Var = {0, 0};
  bar(Var);
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}

