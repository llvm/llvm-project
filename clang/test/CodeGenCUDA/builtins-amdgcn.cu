// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:  -aux-triple x86_64-pc-windows-msvc -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: @_Z16use_dispatch_ptrPi(
// CHECK: %[[PTR:.*]] = call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: %{{.*}} = addrspacecast i8 addrspace(4)* %[[PTR]] to i8*
__global__ void use_dispatch_ptr(int* out) {
  const int* dispatch_ptr = (const int*)__builtin_amdgcn_dispatch_ptr();
  *out = *dispatch_ptr;
}

// CHECK-LABEL: @_Z12test_ds_fmaxf(
// CHECK: call contract float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* @_ZZ12test_ds_fmaxfE6shared, float %{{[^,]*}}, i32 0, i32 0, i1 false)
__global__
void test_ds_fmax(float src) {
  __shared__ float shared;
  volatile float x = __builtin_amdgcn_ds_fmaxf(&shared, src, 0, 0, false);
}

// CHECK-LABEL: @_Z6endpgmv(
// CHECK: call void @llvm.amdgcn.endpgm()
__global__ void endpgm() {
  __builtin_amdgcn_endpgm();
}

// Check the 64 bit argument is correctly passed to the intrinsic without truncation or assertion.

// CHECK-LABEL: @_Z14test_uicmp_i64
// CHECK:  store i64* %out, i64** %out.addr.ascast
// CHECK-NEXT:  store i64 %a, i64* %a.addr.ascast
// CHECK-NEXT:  store i64 %b, i64* %b.addr.ascast
// CHECK-NEXT:  %[[V0:.*]] = load i64, i64* %a.addr.ascast
// CHECK-NEXT:  %[[V1:.*]] = load i64, i64* %b.addr.ascast
// CHECK-NEXT:  %[[V2:.*]] = call i64 @llvm.amdgcn.icmp.i64.i64(i64 %0, i64 %1, i32 35)
// CHECK-NEXT:  %[[V3:.*]] = load i64*, i64** %out.addr.ascast
// CHECK-NEXT:  store i64 %[[V2]], i64* %[[V3]]
// CHECK-NEXT:  ret void
__global__ void test_uicmp_i64(unsigned long long *out, unsigned long long a, unsigned long long b)
{
  *out = __builtin_amdgcn_uicmpl(a, b, 30+5);
}

// Check the 64 bit return value is correctly returned without truncation or assertion.

// CHECK-LABEL: @_Z14test_s_memtime
// CHECK: %[[V1:.*]] = call i64 @llvm.amdgcn.s.memtime()
// CHECK-NEXT: %[[PTR:.*]] = load i64*, i64** %out.addr.ascast
// CHECK-NEXT:  store i64 %[[V1]], i64* %[[PTR]]
// CHECK-NEXT:  ret void
__global__ void test_s_memtime(unsigned long long* out)
{
  *out = __builtin_amdgcn_s_memtime();
}
