// RUN: %clang_cc1 -std=c++1z -emit-pch -o %t.ast %S/Inputs/inject1.cpp
// RUN: %clang_cc1 -std=c++1z -emit-obj -o /dev/null -ast-merge %t.ast %S/Inputs/inject2.cpp
// expected-no-diagnostics
