// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/anonymous-fields1.cpp
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/anonymous-fields2.cpp
// RUN: %clang_cc1 -emit-obj -o /dev/null -ast-merge %t.1.ast -ast-merge %t.2.ast %s
// expected-no-diagnostics
