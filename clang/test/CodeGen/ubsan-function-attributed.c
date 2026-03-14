// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -S -triple x86_64 -std=c17 -fsanitize=function %s -o - | FileCheck %s --check-prefixes=CHECK

// CHECK: .long	248076293
void __attribute__((ms_abi)) f(void) {}

// CHECK: .long	905068220
void g(void) {}

// CHECK: .long	1717976574
void __attribute__((ms_abi)) f_no_prototype() {}

// CHECK: .long	1717976574
void g_no_prototype() {}
