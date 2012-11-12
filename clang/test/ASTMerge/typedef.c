// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/typedef1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/typedef2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: typedef2.c:4:10: error: external variable 'x2' declared with incompatible types in different translation units ('Typedef2' (aka 'double') vs. 'Typedef2' (aka 'int'))
// CHECK: typedef1.c:4:10: note: declared here with type 'Typedef2' (aka 'int')
// CHECK: 1 error
