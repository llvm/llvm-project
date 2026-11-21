// Test target codegen - host bc file has to be created first.
//
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s -check-prefixes=CHECK,CHECK-NVPTX
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -mllvm -openmp-ir-builder-use-default-max-threads=false -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -check-prefix=CHECK,CHECK-NVPTX-FLAG %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo(int N) {
// CHECK: define {{.*}}l12{{.*}} #[[ATTR0:[0-9]+]]
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i)
    ;
// CHECK: define {{.*}}l16{{.*}} #[[ATTR1:[0-9]+]]
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i)
    ;

// CHECK: define {{.*}}l21{{.*}} #[[ATTR2:[0-9]+]]
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42))))
  for (int i = 0; i < N; ++i)
    ;

// CHECK: define {{.*}}l26{{.*}} #[[ATTR3:[0-9]+]]
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42)))) num_threads(22)
  for (int i = 0; i < N; ++i)
    ;
}

#endif

// CHECK-NVPTX: attributes #[[ATTR0]] = {{{.*}} "nvvm.maxntid"="128" {{.*}}}
// CHECK-NVPTX: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxntid"="4" {{.*}}}
// CHECK-NVPTX: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="42" {{.*}}}
// CHECK-NVPTX: attributes #[[ATTR3]] = {{{.*}} "nvvm.maxntid"="22" {{.*}}}

// CHECK-NVPTX-FLAG: attributes #[[ATTR0]] = {
// CHECK-NVPTX-FLAG-NOT: nvvm.maxntid
// CHECK-NVPTX-FLAG-SAME: }
// CHECK-NVPTX-FLAG: attributes #[[ATTR1]] = {{{.*}} "nvvm.maxntid"="4" {{.*}}}
// CHECK-NVPTX-FLAG: attributes #[[ATTR2]] = {{{.*}} "nvvm.maxntid"="42" {{.*}}}
// CHECK-NVPTX-FLAG: attributes #[[ATTR3]] = {{{.*}} "nvvm.maxntid"="22" {{.*}}}
