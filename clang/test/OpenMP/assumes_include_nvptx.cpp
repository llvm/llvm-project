// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown  -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -fexceptions -fcxx-exceptions -aux-triple powerpc64le-unknown-unknown -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#include <cmath>

// TODO: Think about teaching the OMPIRBuilder about default attributes as well so the __kmpc* declarations are annotated.

// CHECK: define internal void @__omp_offloading_{{.*}}__Z17complex_reductionIfEvv_{{.*}}_worker() [[attr0:#[0-9]*]]
// CHECK: define weak void @__omp_offloading_{{.*}}__Z17complex_reductionIfEvv_{{.*}}() [[attr0]]
// CHECK: %call = call float @_Z3sinf(float 0.000000e+00) [[attr5:#[0-9]*]]
// CHECK-DAG: declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() [[attr1:#[0-9]*]]
// CHECK-DAG: declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() [[attr1]]
// CHECK-DAG: declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() [[attr1]]
// CHECK: declare void @__kmpc_kernel_init(i32, i16)
// CHECK-NOT: #
// CHECK: declare void @__kmpc_data_sharing_init_stack()
// CHECK-NOT: #
// CHECK: declare float @_Z3sinf(float) [[attr2:#[0-9]*]]
// CHECK: declare void @__kmpc_kernel_deinit(i16)
// CHECK-NOT: #
// CHECK: declare void @__kmpc_barrier_simple_spmd(%struct.ident_t*, i32) [[attr3:#[0-9]*]]
// CHECK: declare i1 @__kmpc_kernel_parallel(i8**)
// CHECK-NOT: #
// CHECK: declare i32 @__kmpc_global_thread_num(%struct.ident_t*) [[attr4:#[0-9]*]]
// CHECK: declare void @__kmpc_kernel_end_parallel()
// CHECK-NOT: #
// CHECK: define internal void @__omp_offloading_{{.*}}__Z17complex_reductionIdEvv_{{.*}}_worker() [[attr0]]
// CHECK: define weak void @__omp_offloading_{{.*}}__Z17complex_reductionIdEvv_{{.*}}() [[attr0]]
// CHECK: %call = call double @_Z3sind(double 0.000000e+00) [[attr5]]
// CHECK: declare double @_Z3sind(double) [[attr2]]

// CHECK:       attributes [[attr0]]
// CHECK-NOT:  "llvm.assume"
// CHECK:       attributes [[attr1]]
// CHECK-NOT:  "llvm.assume"
// CHECK:       attributes [[attr2]]
// CHECK-SAME:  "llvm.assume"="check_that_this_is_attached_to_included_functions_and_template_instantiations"
// CHECK:       attributes [[attr3]]
// CHECK-NOT:  "llvm.assume"
// CHECK:       attributes [[attr4]]
// CHECK-NOT:  "llvm.assume"
// CHECK:       attributes [[attr5]]
// CHECK-SAME:  "llvm.assume"="check_that_this_is_attached_to_included_functions_and_template_instantiations"


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
