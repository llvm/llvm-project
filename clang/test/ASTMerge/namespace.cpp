// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/namespace1.cpp
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/namespace2.cpp
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: namespace2.cpp:16:17: error: external variable 'z' declared with incompatible types in different translation units ('double' vs. 'float')
// CHECK: namespace1.cpp:16:16: note: declared here with type 'float'
