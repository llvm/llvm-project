// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s \
// RUN:   -Wno-undefined-internal 2>&1 | FileCheck  %s
//
// Test that the "s390x-visible-vector-ABI" module flag is not emitted.

// Calling *local* function with vector argument.

typedef __attribute__((vector_size(16))) int v4i32;

static void bar(v4i32 arg);

void foo() {
  v4i32 Var = {0, 0, 0, 0};
  bar(Var);
}

//CHECK-NOT: !{i32 2, !"s390x-visible-vector-ABI", i32 1}
