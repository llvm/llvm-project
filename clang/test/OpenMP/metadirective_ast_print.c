// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=CHECK-AMDGCN

// RUN: %clang_cc1 -verify -fopenmp-simd -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=CHECK-AMDGCN
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void bar(void);

#define N 10
void foo(void) {
#pragma omp metadirective when(device = {kind(cpu)} \
                               : parallel) default()
  bar();
#pragma omp metadirective when(implementation = {vendor(score(0)  \
                                                        : llvm)}, \
                               device = {kind(cpu)}               \
                               : parallel) default(target teams)
  bar();
#pragma omp metadirective when(device = {kind(gpu)}                                 \
                               : target teams) when(implementation = {vendor(llvm)} \
                                                    : parallel) default()
  bar();
#pragma omp metadirective default(target) when(implementation = {vendor(score(5)  \
                                                                        : llvm)}, \
                                               device = {kind(cpu, host)}         \
                                               : parallel)
  bar();
#pragma omp metadirective when(user = {condition(N > 10)}                 \
                               : target) when(user = {condition(N == 10)} \
                                              : parallel)
  bar();
#pragma omp metadirective when(device = {kind(host)} \
                               : parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_none)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;

// Test metadirective with nested OpenMP directive.
  int array[16];
  #pragma omp metadirective when(user = {condition(1)} \
                                 : parallel for)
  for (int i = 0; i < 16; i++) {
    #pragma omp simd
    for (int j = 0; j < 16; j++)
      array[i] = i;
  }

#pragma omp metadirective when(device={arch("amdgcn")}: \
                                teams distribute parallel for)\
                                default(parallel for)
  for (int i = 0; i < 100; i++)
  ;
}

// CHECK: void bar(void);
// CHECK: void foo(void)
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: bar()
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: bar()
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: bar()
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: bar()
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: bar()
// CHECK-NEXT: #pragma omp parallel for
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK: #pragma omp parallel
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK: #pragma omp parallel for
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK: #pragma omp parallel
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK: #pragma omp parallel for
// CHECK-NEXT: for (int i = 0; i < 16; i++) {
// CHECK-NEXT: #pragma omp simd
// CHECK-NEXT: for (int j = 0; j < 16; j++)
// CHECK-AMDGCN: #pragma omp teams distribute parallel for
// CHECK-AMDGCN-NEXT: for (int i = 0; i < 100; i++)

#endif
