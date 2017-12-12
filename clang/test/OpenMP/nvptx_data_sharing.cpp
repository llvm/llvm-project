// Test device data sharing codegen.
///==========================================================================///

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK1

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void test_ds(){
  #pragma omp target
  {
    int a = 10;
    #pragma omp parallel
    {
       a = 1000;
    }
  }
}

/// ========= In the worker function ========= ///

// CK1: define internal void @__omp_offloading_{{.*}}test_ds{{.*}}worker() [[ATTR1:#.*]] {
// CK1: [[SHAREDARGS:%.+]] = alloca i8**
// CK1: call i1 @__kmpc_kernel_parallel(i8** %work_fn, i8*** [[SHAREDARGS]])
// CK1: [[SHARGSTMP:%.+]] = load i8**, i8*** [[SHAREDARGS]]
// CK1: call void @__omp_outlined___wrapper{{.*}}({{.*}}, i8** [[SHARGSTMP]])

/// ========= In the kernel function ========= ///

// CK1: {{.*}}define void @__omp_offloading{{.*}}test_ds{{.*}}() [[ATTR2:#.*]] {
// CK1: [[SHAREDARGS1:%.+]] = alloca i8**
// CK1: call void @__kmpc_kernel_prepare_parallel({{.*}}, i8*** [[SHAREDARGS1]], i32 1)
// CK1: [[SHARGSTMP1:%.+]] = load i8**, i8*** [[SHAREDARGS1]]
// CK1: [[SHARGSTMP2:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP1]]
// CK1: [[SHAREDVAR:%.+]] = bitcast i32* {{.*}} to i8*
// CK1: store i8* [[SHAREDVAR]], i8** [[SHARGSTMP2]]

/// ========= In the data sharing wrapper function ========= ///

// CK1: {{.*}}define internal void @__omp_outlined___wrapper({{.*}}i8**) [[ATTR1]] {
// CK1: [[SHAREDARGS2:%.+]] = alloca i8**
// CK1: store i8** %2, i8*** [[SHAREDARGS2]]
// CK1: [[SHARGSTMP3:%.+]] = load i8**, i8*** [[SHAREDARGS2]]
// CK1: [[SHARGSTMP4:%.+]] = getelementptr inbounds i8*, i8** [[SHARGSTMP3]]
// CK1: [[SHARGSTMP5:%.+]] = bitcast i8** [[SHARGSTMP4]] to i32**
// CK1: [[SHARGSTMP6:%.+]] = load i32*, i32** [[SHARGSTMP5]]
// CK1: call void @__omp_outlined__({{.*}}, i32* [[SHARGSTMP6]])

/// ========= Attributes ========= ///

// CK1-NOT: attributes [[ATTR1]] = { {{.*}}"has-nvptx-shared-depot"{{.*}} }
// CK1: attributes [[ATTR2]] = { {{.*}}"has-nvptx-shared-depot"{{.*}} }

#endif
