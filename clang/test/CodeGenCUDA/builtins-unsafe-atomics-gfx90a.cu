// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

#define __device__ __attribute__((device))
typedef __attribute__((address_space(3))) float *LP;

// CHECK-LABEL: test_ds_atomic_add_f32
// CHECK: %[[ADDR_ADDR:.*]] = alloca float*, align 8, addrspace(5)
// CHECK: %[[ADDR_ADDR_ASCAST_PTR:.*]] = addrspacecast float* addrspace(5)* %[[ADDR_ADDR]] to float**
// CHECK: store float* %addr, float** %[[ADDR_ADDR_ASCAST_PTR]], align 8
// CHECK: %[[ADDR_ADDR_ASCAST:.*]] = load float*, float** %[[ADDR_ADDR_ASCAST_PTR]], align 8
// CHECK: %[[AS_CAST:.*]] = addrspacecast float* %[[ADDR_ADDR_ASCAST]] to float addrspace(3)*
// CHECK: %3 = call contract float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %[[AS_CAST]]
// CHECK: %4 = load float*, float** %rtn.ascast, align 8
// CHECK: store float %3, float* %4, align 4
__device__ void test_ds_atomic_add_f32(float *addr, float val) {
  float *rtn;
  *rtn = __builtin_amdgcn_ds_faddf((LP)addr, val, 0, 0, 0);
}
