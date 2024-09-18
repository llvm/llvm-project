// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++2c %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s

// expected-no-diagnostics

namespace cwg563 { // cwg563: 3.3
extern "C" int a;
} // namespace cwg563

// CHECK:      LinkageSpecDecl {{.*}} C
// CHECK-NEXT: `-VarDecl {{.*}} a 'int'
