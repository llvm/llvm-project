// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#if FB_NULLIFY
#define FALLBACK_MODIFIER fb_nullify:
#elif FB_PRESERVE
#define FALLBACK_MODIFIER fb_preserve:
#else
#define FALLBACK_MODIFIER
#endif

void f1(void *);

///==========================================================================///
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK1 -DFB_NULLIFY=1 | FileCheck %s --check-prefix=CK1-NULLIFY
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK1 -DFB_PRESERVE=1 | FileCheck %s --check-prefix=CK1-PRESERVE
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK1 | FileCheck %s --check-prefix=CK1-DEFAULT
#ifdef CK1
// CK1-NULLIFY: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x8040]]]
// CK1-PRESERVE: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]
// CK1-DEFAULT: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]
void f2(int *p) {
#pragma omp target data use_device_ptr(FALLBACK_MODIFIER p)
  {
    f1(p);
  }
}
#endif

///==========================================================================///
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK2 -DFB_NULLIFY=1 | FileCheck %s --check-prefix=CK2-NULLIFY
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK2 -DFB_PRESERVE=1 | FileCheck %s --check-prefix=CK2-PRESERVE
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK2 | FileCheck %s --check-prefix=CK2-DEFAULT
#ifdef CK2
// CK2-NULLIFY: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x8043]], i64 [[#0x4000]]]
// CK2-PRESERVE: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x43]], i64 [[#0x4000]]]
// CK2-DEFAULT: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x43]], i64 [[#0x4000]]]
void f2(int *p) {
#pragma omp target data map(p[0:10]) use_device_ptr(FALLBACK_MODIFIER p)
  {
    f1(p);
  }
}
#endif

///==========================================================================///
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK3 -DFB_NULLIFY=1 | FileCheck %s --check-prefix=CK3-NULLIFY
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK3 -DFB_PRESERVE=1 | FileCheck %s --check-prefix=CK3-PRESERVE
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK3 | FileCheck %s --check-prefix=CK3-DEFAULT
#ifdef CK3
// CK3-NULLIFY: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x8040]]]
// CK3-PRESERVE: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]
// CK3-DEFAULT: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 [[#0x40]]]
struct S {
  int *p;
  __attribute__((used)) void f2() {
#pragma omp target data use_device_ptr(FALLBACK_MODIFIER p)
    {
      f1(p);
    }
  }
};
#endif

///==========================================================================///
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK4 -DFB_NULLIFY=1 | FileCheck %s --check-prefix=CK4-NULLIFY
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK4 -DFB_PRESERVE=1 | FileCheck %s --check-prefix=CK4-PRESERVE
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping -DCK4 | FileCheck %s --check-prefix=CK4-DEFAULT
#ifdef CK4
// CK4-NULLIFY: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x8043]], i64 [[#0x4000]]]
// CK4-PRESERVE: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x43]], i64 [[#0x4000]]]
// CK4-DEFAULT: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 [[#0x43]], i64 [[#0x4000]]]
struct S {
  int *p;
  __attribute__((used)) void f2() {
#pragma omp target data map(p[0:10]) use_device_ptr(FALLBACK_MODIFIER p)
    {
      f1(p);
    }
  }
};
#endif
#endif
