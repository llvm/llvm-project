// REQUIRES: amdgpu-registered-target

// RUN: %clang -fopenmp --no-offload-new-driver -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -emit-llvm -S %s -o - | FileCheck %s -check-prefix=LL
// RUN: %clang -fopenmp --no-offload-new-driver -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -S %s -o - | FileCheck %s -check-prefix=S

#ifndef HEADER
#define HEADER

#include <stdio.h>

// CHECK: __CLANG_OFFLOAD_BUNDLE____START__ openmp-amdgcn-amd-amdhsa-gfx90a
// LL: target triple = "amdgcn-amd-amdhsa"
// S: .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
// CHECK: __CLANG_OFFLOAD_BUNDLE____START__ host-
int veccopy() {
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
  a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  #pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}
#endif
