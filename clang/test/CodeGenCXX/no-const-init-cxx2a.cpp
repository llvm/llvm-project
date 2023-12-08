// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -std=c++2a | FileCheck %s

// CHECK-DAG: @p = {{.*}} null
// CHECK-DAG: @_ZGR1p_ = {{.*}} null
int *const &p = new int;

struct d {
  constexpr d(int &&f) : e(f) {}
  int &e;
};

// CHECK-DAG: @g = {{.*}} null
// CHECK-DAG: @_ZGR1g_ = {{.*}} zeroinitializer
d &&g{{0}};

// CHECK: define {{.*}} @__cxx_global_var_init
// CHECK: define {{.*}} @__cxx_global_var_init
// CHECK-NOT: define {{.*}} @__cxx_global_var_init
