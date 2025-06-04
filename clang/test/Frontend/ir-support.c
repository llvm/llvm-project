// Test that we can consume LLVM IR/bitcode in the frontend and produce
// equivalent output to a standard compilation.

// We strip differing '.file' directives before comparing.

// Reference output:
// RUN: %clang_cc1 -triple=x86_64-windows -S -o - %s | grep -v '\.file' > %t.s

// LLVM bitcode:
// RUN: %clang_cc1 -triple=x86_64-windows -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple=x86_64-windows -S -o - %t.bc | grep -v '\.file' > %t.bc.s
// RUN: diff %t.s %t.bc.s

// LLVM IR source code:
// RUN: %clang_cc1 -triple=x86_64-windows -emit-llvm -o %t.ll %s
// RUN: %clang_cc1 -triple=x86_64-windows -S -o - %t.ll | grep -v '\.file' > %t.ll.s
// RUN: diff %t.s %t.ll.s

int f(void) { return 0; }
