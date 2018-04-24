// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang_cc1 -Qn -emit-llvm -debug-info-kind=limited -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-QN %s
// RUN: %clang_cc1 -Qy -emit-llvm -debug-info-kind=limited -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-QY %s

// CHECK-NONE: @main
// CHECK-NONE: llvm.ident
// CHECK-NONE: producer:

// CHECK-QN: @main
// CHECK-QN-NOT: llvm.ident
// CHECK-QN-NOT: producer:

// CHECK-QY: @main
// CHECK-QY: llvm.ident
// CHECK-QY: producer:
int main(void) {}
