// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -disable-llvm-passes -emit-llvm -verify %s -o - | FileCheck %s

// This test checks that the sycl_external attribute is ignored for variadic
// functions; no device code is emitted for such functions.

// expected-warning@+1{{'clang::sycl_external' attribute ignored; a variadic function cannot be called from device code}}
[[clang::sycl_external]] int variadic(int n, ...) { return n; }
// CHECK-NOT: @_Z8variadiciz

int reachedFromVariadicOnly() { return 1; }
// CHECK-NOT: @_Z23reachedFromVariadicOnlyv

// expected-warning@+1{{'clang::sycl_external' attribute ignored; a variadic function cannot be called from device code}}
[[clang::sycl_external]] int variadicCaller(int n, ...) {
  return reachedFromVariadicOnly();
}
// CHECK-NOT: @_Z14variadicCalleriz

// CHECK: define dso_local spir_func noundef i32 @_Z11nonVariadici
[[clang::sycl_external]] int nonVariadic(int n) { return n; }
