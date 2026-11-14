// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -disable-llvm-passes -emit-llvm -verify %s -o - | FileCheck %s \
// RUN:   --implicit-check-not=variadic_fn_test1 \
// RUN:   --implicit-check-not=reached_from_variadic_fn_test2 \
// RUN:   --implicit-check-not=variadic_caller_fn_test3

// This test checks that the sycl_external attribute is ignored for variadic
// functions; no device code is emitted for such functions.

// expected-warning@+1{{'clang::sycl_external' attribute ignored; a variadic function cannot be called from device code}}
[[clang::sycl_external]] int variadic_fn_test1(int n, ...) { return n; }

int reached_from_variadic_fn_test2() { return 1; }

// expected-warning@+1{{'clang::sycl_external' attribute ignored; a variadic function cannot be called from device code}}
[[clang::sycl_external]] int variadic_caller_fn_test3(int n, ...) {
  return reached_from_variadic_fn_test2();
}

// Check that non-variadic functions are emitted to ensure the lack of emission
// for the others is due to their being variadic functions.
[[clang::sycl_external]] int non_variadic_fn_test4(int n) { return n; }
// CHECK: define dso_local spir_func noundef i32 @{{.*}}non_variadic_fn_test4
