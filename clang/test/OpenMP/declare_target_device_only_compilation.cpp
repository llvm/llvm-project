//==========================================================================///
// RUN: %clang -S -target powerpc64le-ibm-linux-gnu -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -S -target i386-pc-linux-gnu -fopenmp -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -S -target x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#pragma omp declare target
#pragma omp begin declare variant match(device={kind(nohost)})
int G1;
#pragma omp end declare variant
#pragma omp end declare target

// CHECK: @[[G:.+]] = hidden {{.*}}global i32 0, align 4
// CHECK: !omp_offload.info = !{!0}
// CHECK: !0 = !{i32 1, !"[[G]]", i32 0, i32 0}
