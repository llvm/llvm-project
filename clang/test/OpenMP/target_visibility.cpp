// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -o - | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -o - | FileCheck %s
// expected-no-diagnostics


#pragma omp declare target

struct A {
void foo() {}
static void sfoo() {}
};

#pragma omp end declare target

struct B {
void bar();
static void sbar();
};

void B::bar() { A a; a.foo(); }
void B::sbar() { A::sfoo(); }
#pragma omp declare target to(B::bar, B::sbar)

[[gnu::visibility("hidden")]] extern const int x = 0;
#pragma omp declare target to(x) device_type(nohost)

[[gnu::visibility("hidden")]] int y = 0;
#pragma omp declare target to(y)

// CHECK-DAG: @x = hidden{{.*}} constant i32 0
// CHECK-DAG: @y = protected{{.*}} i32 0
// CHECK-DAG: define hidden void @_ZN1B4sbarEv()
// CHECK-DAG: define linkonce_odr hidden void @_ZN1A4sfooEv()
// CHECK-DAG: define hidden void @_ZN1B3barEv(
// CHECK-DAG: define linkonce_odr hidden void @_ZN1A3fooEv(
