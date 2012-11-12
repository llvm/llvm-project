// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: ; ModuleID
template<typename> struct A { static int a; };

// CHECK-NOT: @_ZN1AIcE1aE
template<> int A<char>::a;

// CHECK: @_ZN1AIbE1aE = global i32 10
template<> int A<bool>::a = 10;


