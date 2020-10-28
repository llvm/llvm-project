// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown  -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -triple nvptx-unknown-unknown -aux-triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -O1 -disable-llvm-optzns -verify -fopenmp -internal-isystem %S/../Headers/Inputs/include -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -fexceptions -fcxx-exceptions -aux-triple powerpc64le-unknown-unknown -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#include <complex>

// Verify we do not add tbaa metadata to type punned memory operations:

// CHECK:      call i64 @__kmpc_shuffle_int64(
// CHECK-NEXT: store i64 %{{.*}}, i64* %{{.*}}, align {{[0-9]+$}}

// CHECK:      call i64 @__kmpc_shuffle_int64(
// CHECK-NEXT: store i64 %{{.*}}, i64* %{{.*}}, align {{[0-9]+$}}

template <typename T>
void complex_reduction() {
#pragma omp target teams distribute
  for (int ib = 0; ib < 100; ib++) {
    std::complex<T> partial_sum;
    const int istart = ib * 4;
    const int iend = (ib + 1) * 4;
#pragma omp parallel for reduction(+ \
                                   : partial_sum)
    for (int i = istart; i < iend; i++)
      partial_sum += std::complex<T>(i, i);
  }
}

void test() {
  complex_reduction<float>();
  complex_reduction<double>();
}
#endif
