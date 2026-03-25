// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1200 \
// RUN:    -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// CHECK-LABEL: @test_int
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 %b)
void test_int(global int* out, int a, int b) {
  *out = __builtin_amdgcn_ds_bpermute(a, b);
}

// CHECK-LABEL: @test_float
// CHECK: [[BC:%.*]] = bitcast float %b to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[BC]])
void test_float(global float* out, int a, float b) {
  *out = __builtin_amdgcn_ds_bpermute(a, b);
}

// CHECK-LABEL: @test_long
// CHECK: [[LO:%.*]] = trunc i64 %b to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[LO]])
// CHECK: [[SHR:%.*]] = lshr i64 %b, 32
// CHECK: [[HI:%.*]] = trunc {{.*}}i64 [[SHR]] to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[HI]])
void test_long(global long* out, int a, long b) {
  *out = __builtin_amdgcn_ds_bpermute(a, b);
}

// CHECK-LABEL: @test_double
// CHECK: [[BC:%.*]] = bitcast double %b to i64
// CHECK: [[LO:%.*]] = trunc i64 [[BC]] to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[LO]])
// CHECK: [[SHR:%.*]] = lshr i64 [[BC]], 32
// CHECK: [[HI:%.*]] = trunc {{.*}}i64 [[SHR]] to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[HI]])
void test_double(global double* out, int a, double b) {
  *out = __builtin_amdgcn_ds_bpermute(a, b);
}

// Global pointer: 64-bit (address space 1), split into 2 words
// CHECK-LABEL: @test_global_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr addrspace(1) %b to i64
// CHECK: [[LO:%.*]] = trunc i64 [[P2I]] to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[LO]])
// CHECK: [[SHR:%.*]] = lshr i64 [[P2I]], 32
// CHECK: [[HI:%.*]] = trunc {{.*}}i64 [[SHR]] to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[HI]])
void test_global_ptr(global long* out, int a, global int* b) {
  global int* res = __builtin_amdgcn_ds_bpermute(a, b);
  *out = (long)res;
}

// Local pointer: 32-bit (address space 3), single bpermute
// CHECK-LABEL: @test_local_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr addrspace(3) %b to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[P2I]])
// CHECK-NOT: lshr
void test_local_ptr(global int* out, int a, local int* b) {
  local int* res = __builtin_amdgcn_ds_bpermute(a, b);
  *out = (int)(long)res;
}

// Private pointer: 32-bit (address space 5), single bpermute
// CHECK-LABEL: @test_private_ptr
// CHECK: [[P2I:%.*]] = ptrtoint ptr addrspace(5) %b to i32
// CHECK: {{.*}}call i32 @llvm.amdgcn.ds.bpermute(i32 %a, i32 [[P2I]])
// CHECK-NOT: lshr
void test_private_ptr(global int* out, int a, private int* b) {
  private int* res = __builtin_amdgcn_ds_bpermute(a, b);
  *out = (int)(long)res;
}
