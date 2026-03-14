// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test that the "s390x-visible-vector-ABI" module flag is not emitted.

// Call to external function *without* vector argument.

void bar(int arg);

void foo() {
  int Var = 0;
  bar(Var);
}

//CHECK-NOT: !{i32 2, !"s390x-visible-vector-ABI", i32 1}
