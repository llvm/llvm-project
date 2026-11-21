// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void bar();

// construct={parallel} matches inside the parallel region.
// CHECK-LABEL: define internal void @_Z17compile_time_pathv.omp_outlined(
// CHECK-NOT:   __kmpc_masked
// CHECK:       call i32 @__kmpc_single(
// CHECK-NOT:   __kmpc_masked
// CHECK:       ret void
void compile_time_path() {
#pragma omp parallel
  {
#pragma omp metadirective when(construct={parallel}: single) otherwise(masked)
    bar();
  }
}

// Outside a parallel region, construct={parallel} does not match.
// CHECK-LABEL: define {{.*}}void @_Z9no_regionv(
// CHECK-NOT:   __kmpc_single
// CHECK:       call i32 @__kmpc_masked(
// CHECK:       ret void
void no_region() {
#pragma omp metadirective when(construct={parallel}: single) otherwise(masked)
  bar();
}
