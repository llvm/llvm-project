// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - \
// RUN:   | FileCheck %s

int Data[8][4];

void section_2d(int P, int I) {
#pragma omp target map(tofrom : Data[P : 1][0 : 4]) firstprivate(P, I)
  {
    Data[P][I] = 1;
  }
}

// CHECK-LABEL: define {{.*}}@{{__omp_offloading_.+}}section_2d
// CHECK-NOT: getelementptr inbounds{{.*}}[8 x [4 x i32]]
// CHECK: getelementptr{{.*}}[8 x [4 x i32]]
// CHECK: getelementptr inbounds{{.*}}[4 x i32]
