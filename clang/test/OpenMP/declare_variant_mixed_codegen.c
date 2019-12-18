// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=50 %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=50 | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=50
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-version=50 | FileCheck %s --check-prefix GPU
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t -fopenmp-version=50
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - -fopenmp-version=50 | FileCheck %s --check-prefix GPU
// expected-no-diagnostics

// HOST: @base = alias i32 (double), i32 (double)* @hst
#ifndef HEADER
#define HEADER

int dev(double i) { return 0; }

int hst(double i) { return 1; }

#pragma omp declare variant(hst) match(device = {kind(host)})
#pragma omp declare variant(dev) match(device = {kind(gpu)})
int base();

// HOST-LABEL: define void @foo()
// HOST: call i32 (double, ...) bitcast (i32 (double)* @base to i32 (double, ...)*)(double -1.000000e+00)
// HOST: call i32 @hst(double -2.000000e+00)
// HOST: call void [[OFFL:@.+_foo_l29]]()
void foo() {
  base(-1);
  hst(-2);
#pragma omp target
  {
    base(-3);
    dev(-4);
  }
}

// HOST: define {{.*}}void [[OFFL]]()
// HOST: call i32 (double, ...) bitcast (i32 (double)* @base to i32 (double, ...)*)(double -3.000000e+00)
// HOST: call i32 @dev(double -4.000000e+00)

// GPU: define {{.*}}void @__omp_offloading_{{.+}}_foo_l29()
// GPU: call i32 @base(double -3.000000e+00)
// GPU: call i32 @dev(double -4.000000e+00)

// GPU: define {{.*}}i32 @base(double
// GPU: ret i32 0
// GPU: define {{.*}}i32 @dev(double
// GPU: ret i32 0

#endif // HEADER
