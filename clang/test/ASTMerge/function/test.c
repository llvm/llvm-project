// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/function1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/function2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only -verify %s

// CHECK: function2.c:3:6: warning: external function 'f1' declared with incompatible types in different translation units ('void (Int, double)' (aka 'void (int, double)') vs. 'void (int, float)')
// CHECK: function1.c:2:6: note: declared here with type 'void (int, float)'
// CHECK: function2.c:5:6: warning: external function 'f3' declared with incompatible types in different translation units ('void (int)' vs. 'void (void)')
// CHECK: function1.c:4:6: note: declared here with type 'void (void)'
// CHECK: 2 warnings generated

// expected-warning@Inputs/function2.c:3 {{external function 'f1' declared with incompatible types}}
// expected-note@Inputs/function1.c:2 {{declared here}}
// expected-warning@Inputs/function2.c:5 {{external function 'f3' declared with incompatible types}}
// expected-note@Inputs/function1.c:4 {{declared here}}
