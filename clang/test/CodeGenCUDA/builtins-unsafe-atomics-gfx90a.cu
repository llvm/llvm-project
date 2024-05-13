// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s \
// RUN:  -o - | FileCheck %s

#define __device__ __attribute__((device))
typedef __attribute__((address_space(3))) float *LP;

// CHECK-LABEL: test_ds_atomic_add_f32
// CHECK: %[[ADDR_ADDR:.*]] = alloca ptr, align 8, addrspace(5)
// CHECK: %[[ADDR_ADDR_ASCAST_PTR:.*]] = addrspacecast ptr addrspace(5) %[[ADDR_ADDR]] to ptr
// CHECK: store ptr %addr, ptr %[[ADDR_ADDR_ASCAST_PTR]], align 8
// CHECK: %[[ADDR_ADDR_ASCAST:.*]] = load ptr, ptr %[[ADDR_ADDR_ASCAST_PTR]], align 8
// CHECK: %[[AS_CAST:.*]] = addrspacecast ptr %[[ADDR_ADDR_ASCAST]] to ptr addrspace(3)
// CHECK: %3 = call contract float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %[[AS_CAST]]
// CHECK: %4 = load ptr, ptr %rtn.ascast, align 8
// CHECK: store float %3, ptr %4, align 4
__device__ void test_ds_atomic_add_f32(float *addr, float val) {
  float *rtn;
  *rtn = __builtin_amdgcn_ds_faddf((LP)addr, val, 0, 0, 0);
}
