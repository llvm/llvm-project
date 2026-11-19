// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Check a dispatch construct in a target region, compiled for the device.

#pragma omp declare target
int f_disp(int);
int f_noctx(int);
#pragma omp declare variant(f_disp) match(construct={dispatch})
#pragma omp declare variant(f_noctx) match(user={condition(1)})
int f(int);
#pragma omp end declare target

// CHECK-LABEL: define {{.*}}amdgpu_kernel void @__omp_offloading_{{.*}}_Z3tgtibb_l{{[0-9]+}}(
// CHECK: br i1
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z1fi(
// CHECK-NEXT: store i32 [[R]], ptr [[OUT:%[^ ,]+]]
// CHECK: br i1
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z7f_noctxi(
// CHECK-NEXT: store i32 [[R]], ptr [[OUT]]
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z6f_dispi(
// CHECK-NEXT: store i32 [[R]], ptr [[OUT]]
void tgt(int a, bool c1, bool c2) {
  int out = 0;
#pragma omp target map(tofrom: out)
  {
#pragma omp dispatch novariants(c1) nocontext(c2)
    out = f(a);
  }
}
