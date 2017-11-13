// Test to ensure -fdebug-pass-manager works when invoking the
// ThinLTO backend path with the new PM.
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -o %t.o -flto=thin -fexperimental-new-pass-manager -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s
// CHECK: Running pass:

void foo() {
}
