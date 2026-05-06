// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -std=c++11 -ast-print %s -o - | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -std=c++11 -ast-print %s -o %t.1.cpp
// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -std=c++11 -ast-print %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp

extern int __attribute__((overflow_behavior(trap))) a;
extern int __attribute__((overflow_behavior(wrap))) b;

extern int __ob_trap c;
extern int __ob_wrap d;

// PRINT: extern __ob_trap int a;
// PRINT: extern __ob_wrap int b;
// PRINT: extern __ob_trap int c;
// PRINT: extern __ob_wrap int d;
