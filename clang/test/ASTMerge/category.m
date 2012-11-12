// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/category1.m
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/category2.m
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: category2.m:18:1: error: instance method 'method2' has incompatible result types in different translation units ('float' vs. 'int')
// CHECK: category1.m:16:1: note: instance method 'method2' also declared here
// CHECK: category2.m:26:1: error: instance method 'method3' has incompatible result types in different translation units ('float' vs. 'int')
// CHECK: category1.m:24:1: note: instance method 'method3' also declared here
// CHECK: category2.m:48:1: error: instance method 'blah' has incompatible result types in different translation units ('int' vs. 'float')
// CHECK: category1.m:46:1: note: instance method 'blah' also declared here
// CHECK: 3 errors generated.
