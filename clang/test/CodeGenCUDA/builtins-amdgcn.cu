// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -x hip \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -x hip \
// RUN:  -aux-triple x86_64-pc-windows-msvc -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: @_Z16use_dispatch_ptrPi(
// CHECK: %[[PTR:.*]] = call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: %{{.*}} = addrspacecast i8 addrspace(4)* %[[PTR]] to i32*
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

// CHECK-LABEL: @_Z12test_ds_faddf(
// CHECK: call contract float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* @_ZZ12test_ds_faddfE6shared, float %{{[^,]*}}, i32 0, i32 0, i1 false)
__global__ void test_ds_fadd(float src) {
  __shared__ float shared;
  volatile float x = __builtin_amdgcn_ds_faddf(&shared, src, 0, 0, false);
}

// CHECK-LABEL: @_Z12test_ds_fminfPf(float %src, float addrspace(1)* %shared.coerce
// CHECK: %shared = alloca float*, align 8, addrspace(5)
// CHECK: %shared.ascast = addrspacecast float* addrspace(5)* %shared to float**
// CHECK: %shared.addr = alloca float*, align 8, addrspace(5)
// CHECK: %shared.addr.ascast = addrspacecast float* addrspace(5)* %shared.addr to float**
// CHECK: %[[S0:.*]] = addrspacecast float addrspace(1)* %shared.coerce to float*
// CHECK: store float* %[[S0]], float** %shared.ascast, align 8
// CHECK: %shared1 = load float*, float** %shared.ascast, align 8
// CHECK: store float* %shared1, float** %shared.addr.ascast, align 8
// CHECK: %[[S1:.*]] = load float*, float** %shared.addr.ascast, align 8
// CHECK: %[[S2:.*]] = addrspacecast float* %[[S1]] to float addrspace(3)*
// CHECK: call contract float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* %[[S2]]
__global__ void test_ds_fmin(float src, float *shared) {
  volatile float x = __builtin_amdgcn_ds_fminf(shared, src, 0, 0, false);
}

// CHECK: @_Z33test_ret_builtin_nondef_addrspace
// CHECK: %[[X:.*]] = alloca i8*, align 8, addrspace(5)
// CHECK: %[[XC:.*]] = addrspacecast i8* addrspace(5)* %[[X]] to i8**
// CHECK: %[[Y:.*]] = call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: %[[YASCAST:.*]] = addrspacecast i8 addrspace(4)* %[[Y]] to i8*
// CHECK: store i8* %[[YASCAST]], i8** %[[XC]], align 8
__device__ void test_ret_builtin_nondef_addrspace() {
  void *x = __builtin_amdgcn_dispatch_ptr();
}

// CHECK-LABEL: @_Z6endpgmv(
// CHECK: call void @llvm.amdgcn.endpgm()
__global__ void endpgm() {
  __builtin_amdgcn_endpgm();
}

// Check the 64 bit argument is correctly passed to the intrinsic without truncation or assertion.

// CHECK-LABEL: @_Z14test_uicmp_i64
// CHECK:  store i64* %out1, i64** %out.addr.ascast
// CHECK-NEXT:  store i64 %a, i64* %a.addr.ascast
// CHECK-NEXT:  store i64 %b, i64* %b.addr.ascast
// CHECK-NEXT:  %[[V0:.*]] = load i64, i64* %a.addr.ascast
// CHECK-NEXT:  %[[V1:.*]] = load i64, i64* %b.addr.ascast
// CHECK-NEXT:  %[[V2:.*]] = call i64 @llvm.amdgcn.icmp.i64.i64(i64 %[[V0]], i64 %[[V1]], i32 35)
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

// Check a generic pointer can be passed as a shared pointer and a generic pointer.
__device__ void func(float *x);

// CHECK: @_Z17test_ds_fmin_funcfPf
// CHECK: %[[SHARED:.*]] = alloca float*, align 8, addrspace(5)
// CHECK: %[[SHARED_ASCAST:.*]] = addrspacecast float* addrspace(5)* %[[SHARED]] to float**
// CHECK: %[[SRC_ADDR:.*]] = alloca float, align 4, addrspace(5)
// CHECK: %[[SRC_ADDR_ASCAST:.*]] = addrspacecast float addrspace(5)* %[[SRC_ADDR]] to float*
// CHECK: %[[SHARED_ADDR:.*]] = alloca float*, align 8, addrspace(5)
// CHECK: %[[SHARED_ADDR_ASCAST:.*]] = addrspacecast float* addrspace(5)* %[[SHARED_ADDR]] to float**
// CHECK: %[[X:.*]] = alloca float, align 4, addrspace(5)
// CHECK: %[[X_ASCAST:.*]] = addrspacecast float addrspace(5)* %[[X]] to float*
// CHECK: %[[SHARED1:.*]] = load float*, float** %[[SHARED_ASCAST]], align 8
// CHECK: store float %src, float* %[[SRC_ADDR_ASCAST]], align 4
// CHECK: store float* %[[SHARED1]], float** %[[SHARED_ADDR_ASCAST]], align 8
// CHECK: %[[ARG0_PTR:.*]] = load float*, float** %[[SHARED_ADDR_ASCAST]], align 8
// CHECK: %[[ARG0:.*]] = addrspacecast float* %[[ARG0_PTR]] to float addrspace(3)*
// CHECK: call contract float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* %[[ARG0]]
// CHECK: %[[ARG0:.*]] = load float*, float** %[[SHARED_ADDR_ASCAST]], align 8
// CHECK: call void @_Z4funcPf(float* %[[ARG0]]) #8
__global__ void test_ds_fmin_func(float src, float *__restrict shared) {
  volatile float x = __builtin_amdgcn_ds_fminf(shared, src, 0, 0, false);
  func(shared);
}
