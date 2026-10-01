// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fno-finite-loops -emit-llvm %s -o %t-nomp.ll
// RUN: FileCheck --input-file=%t-nomp.ll %s -check-prefix=LLVM-NOMP

void foo() {}

// CIR: cir.func {{.*}}@_Z3foov{{.*}}attributes {{.*}}mustprogress

// LLVM: define {{.*}} void @_Z3foov(){{.*}} #[[LATTR:[0-9]+]]
// LLVM: attributes #[[LATTR]] = {{[{].*}}mustprogress{{.*}}}

// With -fno-finite-loops, checkIfFunctionMustProgress() is false, so no
// function should carry 'mustprogress' at all.
// LLVM-NOMP: define {{.*}} void @_Z3foov()
// LLVM-NOMP-NOT: mustprogress
