// RUN: %clang_cc1 -verify -fopenmp -std=c++20 -x c++ -triple x86_64-unknown-unknown \
// RUN:   -Wno-bit-int-extension -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

void one_i32(unsigned n) {
#pragma omp parallel for collapse(1)
  for (unsigned i = 0; i < n; ++i)
    ;
}

// CHECK-LABEL: define internal void @_Z7one_i32j.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_4u(

void one_i40(_BitInt(40) n) {
#pragma omp parallel for collapse(1)
  for (_BitInt(40) i = 0; i < n; ++i)
    ;
}

// CHECK-LABEL: define internal void @_Z7one_i40DB40_.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_8(

void dynamic_two(unsigned n, unsigned m) {
#pragma omp parallel for collapse(2)
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < m; ++j)
      ;
}

// CHECK-LABEL: define internal void @_Z11dynamic_twojj.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_8(

void fit_constant() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j)
      ;
}

// CHECK-LABEL: define internal void @_Z12fit_constantv.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_4(

void wide_constant() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < 100000; ++i)
    for (int j = 0; j < 100000; ++j)
      ;
}

// CHECK-LABEL: define internal void @_Z13wide_constantv.omp_outlined(
// CHECK: call void @__kmpc_for_static_init_8(
