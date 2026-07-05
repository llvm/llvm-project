// RUN: %clang_cc1 -emit-pch -o %t1 -detailed-preprocessing-record %S/Inputs/chain-macro1.h
// RUN: %clang_cc1 -emit-pch -o %t2 -detailed-preprocessing-record %S/Inputs/chain-macro2.h -include-pch %t1
// RUN: %clang_cc1 -fsyntax-only -verify -include-pch %t2 %s
// RUN: %clang_cc1 -ast-print -include-pch %t2 %s | FileCheck %s
// expected-no-diagnostics

// CHECK: void f(void);
FOOBAR
// CHECK: void g(void);
BARFOO
