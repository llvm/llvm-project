// Test to ensure the opt level is passed down to the ThinLTO backend.
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -O2 -o %t.o -flto=thin -fexperimental-new-pass-manager -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fno-experimental-new-pass-manager -mllvm -debug-pass=Structure 2>&1 | FileCheck %s --check-prefix=O2-OLDPM
// O2-OLDPM: Loop Vectorization

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O2 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O2-NEWPM
// O2-NEWPM: Running pass: LoopVectorizePass

// RUN: %clang_cc1 -O0 -o %t.o -flto=thin -fexperimental-new-pass-manager -triple x86_64-unknown-linux-gnu -emit-llvm-bc %s
// RUN: llvm-lto -thinlto -o %t %t.o

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -fexperimental-new-pass-manager 2>&1 | FileCheck %s --check-prefix=O0-NEWPM
// O0-NEWPM-NOT: Running pass: LoopVectorizePass

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -O0 -o %t2.o -x ir %t.o -fthinlto-index=%t.thinlto.bc -fno-experimental-new-pass-manager -mllvm -debug-pass=Structure 2>&1 | FileCheck %s --check-prefix=O0-OLDPM
// O0-OLDPM-NOT: Loop Vectorization

void foo() {
}
