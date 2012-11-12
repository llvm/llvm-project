// RUN: %clang_cc1 -emit-pch -x objective-c-header -o %t1 %S/Inputs/chain-remap-types1.h
// RUN: %clang_cc1 -emit-pch -x objective-c-header -o %t2 %S/Inputs/chain-remap-types2.h -include-pch %t1
// RUN: %clang_cc1 -include-pch %t2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -ast-print -include-pch %t2 %s | FileCheck %s
// expected-no-diagnostics

// CHECK: @class X;
// CHECK: struct Y 
// CHECK: @property ( assign,readwrite,atomic ) X * prop
// CHECK: void h(X *);
// CHECK: @interface X(Blah)
// CHECK: void g(X *);

