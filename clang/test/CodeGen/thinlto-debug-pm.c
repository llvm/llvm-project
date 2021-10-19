// Test to ensure the opt level is passed down to the ThinLTO backend.
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -O2 -o %t.o -flto=thin -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager 2>&1 | FileCheck %s --check-prefix=O2
// O2: Running pass: LoopVectorizePass

// RUN: %clang_cc1 -O0 -o %t.o -flto=thin -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager 2>&1 | FileCheck %s --check-prefix=O0
// O0-NOT: Running pass: LoopVectorizePass

void foo() {
}
