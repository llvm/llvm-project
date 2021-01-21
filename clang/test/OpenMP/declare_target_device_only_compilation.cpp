// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-pc-linux-gnu -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-i386-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-i386-host.bc -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-x86_64-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86_64-host.bc -o - | FileCheck %s

// expected-no-diagnostics

#pragma omp declare target
#pragma omp begin declare variant match(device={kind(nohost)})
int G1;
#pragma omp end declare variant
#pragma omp end declare target

// CHECK: @[[G:.+]] = hidden {{.*}}global i32 0, align 4
// CHECK: !omp_offload.info = !{!0}
// CHECK: !0 = !{i32 1, !"[[G]]", i32 0, i32 0}
