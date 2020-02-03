// RUN: %clang_cc1 -std=c++1y -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// CHECK: @x = global {{.*}} zeroinitializer

// CHECK: define {{.*}} @_Z1fv
inline auto f() {
  int n = 0;
  // CHECK: load i32
  // CHECK: store i32
  // CHECK: ret
  return [=] () mutable { return ++n; };
}

auto x = f();

template<typename T> auto *g(T t) { return t; }
template<typename T> decltype(auto) h(T t) { return t; }

// CHECK: define {{.*}} @_Z1zv
void z() {
  // CHECK: call {{.*}} @_Z1gIPZ1fvEUlvE_EPDaT_(
  // CHECK: call {{.*}} @_Z1hIPZ1fvEUlvE_EDcT_(
  g(&x);
  h(&x);
}

auto i() { return [] {}; }
// CHECK: define {{.*}} @_Z1jv
auto j() {
  // CHECK: call {{.*}} @"_Z1hIZ1ivE3$_0EDcT_"()
  h(i());
}
