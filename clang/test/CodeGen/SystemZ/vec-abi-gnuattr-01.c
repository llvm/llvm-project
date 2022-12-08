// RUN: %clang_cc1 -triple s390x-ibm-linux -target-cpu arch13 -emit-llvm \
// RUN:   -fzvector -o - %s 2>&1 | FileCheck  %s
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// Call to external function with vector return value.

typedef __attribute__((vector_size(16))) int v4i32;

v4i32 bar(void);

void foo(v4i32 *Dst) {
  v4i32 Var = bar();
  *Dst = Var;
}

//CHECK: !llvm.module.flags = !{!0, !1}
//CHECK: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}
