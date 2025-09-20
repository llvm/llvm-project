// RUN: %clang_cc1 -foverflow-behavior-types -std=c++11 -ast-print %s -o - | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -foverflow-behavior-types -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -foverflow-behavior-types -std=c++11 -ast-print %s -o %t.1.cpp
// RUN: %clang_cc1 -foverflow-behavior-types -std=c++11 -ast-print %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp

extern int __attribute__((overflow_behavior(no_wrap))) a;
extern int __attribute__((overflow_behavior(wrap))) b;

// PRINT: extern int a __attribute__((overflow_behavior(no_wrap)));
// PRINT: extern int b __attribute__((overflow_behavior(wrap)));
