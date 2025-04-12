// RUN: %clang_cc1 -ast-print -std=c++2c %s | FileCheck %s

template <class... T, unsigned N>
auto foo(T ...params) {
  return params...[N];
}

// CHECK: template <class ...T, unsigned int N> auto foo(T ...params) {
// CHECK-NEXT: return params...[N];
