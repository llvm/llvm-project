// RUN: %clang_cc1 -triple s390x-ibm-linux -target-cpu arch10 -emit-llvm \
// RUN:   -fzvector -o - %s 2>&1 | FileCheck  %s --check-prefix=MODFLAG
//
// Test the emission of the "s390x-visible-vector-ABI" module flag.

// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-ibm-linux -target-cpu arch10 -S \
// RUN:   -fzvector -o - %s 2>&1 | FileCheck  %s --check-prefix=ARCH10-ASM
// RUN: %clang_cc1 -triple s390x-ibm-linux -target-cpu arch13 -S \
// RUN:   -fzvector -o - %s 2>&1 | FileCheck  %s --check-prefix=ARCH13-ASM
//
// Test the emission of a gnu attribute describing the vector ABI.

// Call to external function with vector argument.

typedef __attribute__((vector_size(16))) int v4i32;

void bar(v4i32 Arg);

void foo() {
  v4i32 Var = {0, 0, 0, 0};
  bar(Var);
}

//MODFLAG: !llvm.module.flags = !{!0, !1}
//MODFLAG: !0 = !{i32 2, !"s390x-visible-vector-ABI", i32 1}

//ARCH10-ASM: .gnu_attribute 8, 1
//ARCH13-ASM: .gnu_attribute 8, 2
