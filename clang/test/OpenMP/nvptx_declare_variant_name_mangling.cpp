// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --implicit-check-not='call i32 {@_Z3bazv|@_Z3barv}'
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --implicit-check-not='call i32 {@_Z3bazv|@_Z3barv}'
// expected-no-diagnostics

// CHECK-DAG: @_Z3barv
// CHECK-DAG: @_Z3bazv
// CHECK-DAG: @"_Z54bar$ompvariant$S2$s8$Pnvptx$Pnvptx64$S3$s10$Pmatch_anyv"
// CHECK-DAG: @"_Z54baz$ompvariant$S2$s8$Pnvptx$Pnvptx64$S3$s10$Pmatch_anyv"
// CHECK-DAG: call i32 @"_Z54bar$ompvariant$S2$s8$Pnvptx$Pnvptx64$S3$s10$Pmatch_anyv"()
// CHECK-DAG: call i32 @"_Z54baz$ompvariant$S2$s8$Pnvptx$Pnvptx64$S3$s10$Pmatch_anyv"()

#ifndef HEADER
#define HEADER

#pragma omp declare target

int bar() { return 1; }

int baz() { return 5; }

#pragma omp begin declare variant match(device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

int bar() { return 2; }

int baz() { return 6; }

#pragma omp end declare variant

#pragma omp end declare target

int main() {
  int res;
#pragma omp target map(from \
                       : res)
  res = bar() + baz();
  return res;
}

#endif