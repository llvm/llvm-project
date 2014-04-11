// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple powerpc-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PPC %s
// RUN: %clang_cc1 -pg -triple powerpc64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PPC %s
// RUN: %clang_cc1 -pg -triple powerpc64le-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PPC %s
void foo(void) {
// CHECK: call void @mcount()
// CHECK-PPC: call void @_mcount()
}
