// RUN: %clang_cc1 -std=c++20 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc | FileCheck %s

// A default argument's lambda is named apart from the function's own lambdas.

template <typename F> int call(F f) { return f(); }

template <int N> int g(int (*f)() = [] { return 10; }) {
  return f() + call([] { return N; });
}

int use() { return g<1>() + g<2>(); }

// CHECK-DAG: define {{.*}} @"??R<lambda_1>@?0???$g@$00@@YAHP6AHXZ@Z@QEBA?A?<auto>@@XZ"(
// CHECK-DAG: define {{.*}} @"??R<lambda_1>@?0???$g@$01@@YAHP6AHXZ@Z@QEBA?A?<auto>@@XZ"(
// CHECK-DAG: define internal {{.*}} @"??R<lambda_1_{{[0-9]+}}>@?0???$g@$00@@YAHP6AHXZ@Z@QEBA?A?<auto>@@XZ"(
// CHECK-DAG: define internal {{.*}} @"??R<lambda_1_{{[0-9]+}}>@?0???$g@$01@@YAHP6AHXZ@Z@QEBA?A?<auto>@@XZ"(
