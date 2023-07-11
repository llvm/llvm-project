// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown  -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -fexceptions -fcxx-exceptions -aux-triple powerpc64le-unknown-unknown -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#include <cmath>

// TODO: Think about teaching the OMPIRBuilder about default attributes as well so the __kmpc* declarations are annotated.

// CHECK: define weak_odr protected void @__omp_offloading_{{.*}}__Z17complex_reductionIfEvv_{{.*}}() [[attr0:#[0-9]]]
// CHECK: call i32 @__kmpc_target_init(
// CHECK: declare noundef float @_Z3sinf(float noundef) [[attr1:#[0-9]*]]
// CHECK: declare void @__kmpc_target_deinit(
// CHECK: define weak_odr protected void @__omp_offloading_{{.*}}__Z17complex_reductionIdEvv_{{.*}}() [[attr0]]
// CHECK: %call = call noundef double @_Z3sind(double noundef 0.000000e+00) [[attr2:#[0-9]]]
// CHECK: declare noundef double @_Z3sind(double noundef) [[attr1]]

// CHECK:       attributes [[attr0]]
// CHECK-NOT:  "llvm.assume"
// CHECK:       attributes [[attr1]]
// CHECK-SAME:  "llvm.assume"="ompx_check_that_this_is_attached_to_included_functions_and_template_instantiations"
// CHECK:       attributes [[attr2]]
// CHECK-SAME:  "llvm.assume"="ompx_check_that_this_is_attached_to_included_functions_and_template_instantiations"


template <typename T>
void foo() {
  cos(T(0));
}

template <typename T>
void complex_reduction() {
  foo<T>();
#pragma omp target
  sin(T(0));
}

#pragma omp assumes ext_check_that_this_is_attached_to_included_functions_and_template_instantiations

void test() {
  complex_reduction<float>();
  complex_reduction<double>();
}
#endif
