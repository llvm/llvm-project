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

// Device weights also count only from the innermost TARGET. KIND is worth 2
// with only TARGET in the context and loses to the vendor score of 3. With
// PARALLEL inside TARGET, it is worth 4 and wins.

#ifdef __NVPTX__
#define KIND gpu
#else
#define KIND cpu
#endif

int kind_outside(void);
int vendor_outside(void);
#pragma omp declare variant(kind_outside) match(device={kind(KIND)})
#pragma omp declare variant(vendor_outside)                                    \
    match(implementation={vendor(score(3): llvm)})
int weight_outside(void);

int kind_inside(void);
int vendor_inside(void);
#pragma omp declare variant(kind_inside) match(device={kind(KIND)})
#pragma omp declare variant(vendor_inside)                                     \
    match(implementation={vendor(score(3): llvm)})
int weight_inside(void);

void outside_weight(void) {
#pragma omp parallel
#pragma omp target
  weight_outside();
}

void inside_weight(void) {
#pragma omp target
#pragma omp parallel
  weight_inside();
}

// CHECK-DAG: call{{.*}} @base_outside(
// CHECK-DAG: call{{.*}} @parallel_inside(
// CHECK-DAG: call{{.*}} @vendor_outside(
// CHECK-DAG: call{{.*}} @kind_inside(
