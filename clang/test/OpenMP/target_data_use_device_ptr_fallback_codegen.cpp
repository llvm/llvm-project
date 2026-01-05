// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DFB_NULLIFY=1 | FileCheck %s --check-prefix=NULLIFY
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DFB_PRESERVE=1 | FileCheck %s --check-prefix=PRESERVE
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix=DEFAULT

// expected-no-diagnostics

// NULLIFY: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x8040]]]
// PRESERVE: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]
// DEFAULT: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]

#ifndef HEADER
#define HEADER

void f1(void *);
void f2(int *p) {
#if FB_NULLIFY
#pragma omp target data use_device_ptr(fb_nullify: p)
#elif FB_PRESERVE
#pragma omp target data use_device_ptr(fb_preserve: p)
#else
#pragma omp target data use_device_ptr(p)
#endif
  {
    f1(p);
  }
}
#endif
