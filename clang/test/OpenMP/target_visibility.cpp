// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -o - | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -o - | FileCheck %s
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

// CHECK-DAG: define hidden void @_ZN1B4sbarEv()
// CHECK-DAG: define linkonce_odr hidden void @_ZN1A4sfooEv()
// CHECK-DAG: define hidden void @_ZN1B3barEv(
// CHECK-DAG: define linkonce_odr hidden void @_ZN1A3fooEv(
