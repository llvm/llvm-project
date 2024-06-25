// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fopenmp-version=45
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-version=45 | FileCheck %s --check-prefix GPU
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t -fopenmp-version=45
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - -fopenmp-version=45 | FileCheck %s --check-prefix GPU

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix GPU
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix GPU
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int dev(double i) { return 0; }

int hst(double i) { return 1; }

#pragma omp declare variant(hst) match(device = {kind(host)})
#pragma omp declare variant(dev) match(device = {kind(gpu)})
int base();

// HOST-LABEL: define{{.*}} void @foo()
// HOST: call i32 @hst(double noundef -1.000000e+00)
// HOST: call i32 @hst(double noundef -2.000000e+00)
// HOST: call void [[OFFL:@.+_foo_l36]]()
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
// HOST: call i32 @hst(double noundef -3.000000e+00)
// HOST: call i32 @dev(double noundef -4.000000e+00)

// GPU: define {{.*}}void @__omp_offloading_{{.+}}_foo_l36(ptr {{[^,]+}})
// GPU: call i32 @dev(double noundef -3.000000e+00)
// GPU: call i32 @dev(double noundef -4.000000e+00)

// GPU-NOT: @base
// GPU: define {{.*}}i32 @dev(double
// GPU: ret i32 0

#endif // HEADER
