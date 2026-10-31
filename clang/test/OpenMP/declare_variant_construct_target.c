// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --implicit-check-not='call{{.*}} @parallel_outside(' --implicit-check-not='call{{.*}} @base_inside('
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --implicit-check-not='call{{.*}} @parallel_outside(' --implicit-check-not='call{{.*}} @base_inside('
// expected-no-diagnostics

// The construct context starts at the innermost enclosing TARGET, so a
// PARALLEL outside the TARGET does not match construct={parallel}.

int parallel_outside(void);
#pragma omp declare variant(parallel_outside) match(construct={parallel})
int base_outside(void);

int parallel_inside(void);
#pragma omp declare variant(parallel_inside) match(construct={parallel})
int base_inside(void);

void outside(void) {
#pragma omp parallel
#pragma omp target
  base_outside();
}

void inside(void) {
#pragma omp parallel
#pragma omp target
#pragma omp parallel
  base_inside();
}

// CHECK-DAG: call{{.*}} @base_outside(
// CHECK-DAG: call{{.*}} @parallel_inside(
