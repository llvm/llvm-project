// Test to ensure -llvm-verify-each works.
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -O2 -o /dev/null -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s -fdebug-pass-manager 2>&1 | FileCheck %s --check-prefix=NO
// NO-NOT: Verifying

// RUN: %clang_cc1 -O2 -o /dev/null -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s -fdebug-pass-manager -llvm-verify-each 2>&1 | FileCheck %s

// RUN: %clang_cc1 -O2 -o %t.o -flto=thin -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s -fdebug-pass-manager -llvm-verify-each 2>&1 | FileCheck %s
// RUN: llvm-lto -thinlto -o %t %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o /dev/null -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -llvm-verify-each 2>&1 | FileCheck %s

// CHECK: Verifying function foo
// CHECK: Running pass: InstCombinePass
// CHECK: Verifying function foo

void foo(void) {
}
