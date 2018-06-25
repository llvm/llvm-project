// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
//
#ifndef HEADER
#define HEADER

// CHECK: @{{.*}}Foo{{.*}}bar{{.*}} = constant i32 1,

// Section A - Define a class with a static constexpr data member.
struct Foo {
  static constexpr int bar = 1;
};

// Section B - ODR-use the data member.
void F(const int &);
void Test() { F(Foo::bar); }

// Section C - Define the data member.
constexpr int Foo::bar;
#endif

