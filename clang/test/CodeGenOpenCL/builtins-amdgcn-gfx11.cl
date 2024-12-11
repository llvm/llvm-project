// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1101 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1102 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1103 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1150 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1151 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1152 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1153 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

typedef unsigned int uint;
typedef unsigned long ulong;
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: @test_s_sendmsg_rtn(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.s.sendmsg.rtn.i32(i32 0)
void test_s_sendmsg_rtn(global uint* out) {
  *out = __builtin_amdgcn_s_sendmsg_rtn(0);
}

// CHECK-LABEL: @test_s_sendmsg_rtnl(
// CHECK: {{.*}}call{{.*}} i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 0)
void test_s_sendmsg_rtnl(global ulong* out) {
  *out = __builtin_amdgcn_s_sendmsg_rtnl(0);
}

// CHECK-LABEL: @test_ds_bvh_stack_rtn(
// CHECK: %0 = tail call{{.*}} { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(i32 %addr, i32 %data, <4 x i32> %data1, i32 128)
// CHECK: %1 = extractvalue { i32, i32 } %0, 0
// CHECK: %2 = extractvalue { i32, i32 } %0, 1
// CHECK: %3 = insertelement <2 x i32> poison, i32 %1, i64 0
// CHECK: %4 = insertelement <2 x i32> %3, i32 %2, i64 1
void test_ds_bvh_stack_rtn(global uint2* out, uint addr, uint data, uint4 data1)
{
  *out = __builtin_amdgcn_ds_bvh_stack_rtn(addr, data, data1, 128);
}

// CHECK-LABEL: @test_permlane64(
// CHECK: {{.*}}call{{.*}} i32 @llvm.amdgcn.permlane64.i32(i32 %a)
void test_permlane64(global uint* out, uint a) {
  *out = __builtin_amdgcn_permlane64(a);
}

// CHECK-LABEL: @test_s_wait_event_export_ready
// CHECK: {{.*}}call{{.*}} void @llvm.amdgcn.s.wait.event.export.ready
void test_s_wait_event_export_ready() {
  __builtin_amdgcn_s_wait_event_export_ready();
}

// CHECK-LABEL: @test_global_add_f32
// CHECK: = atomicrmw fadd ptr addrspace(1) %addr, float %x syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}
#if !defined(__SPIRV__)
void test_global_add_f32(float *rtn, global float *addr, float x) {
#else
void test_global_add_f32(float *rtn, __attribute__((address_space(1))) float *addr, float x) {
#endif
  *rtn = __builtin_amdgcn_global_atomic_fadd_f32(addr, x);
}
