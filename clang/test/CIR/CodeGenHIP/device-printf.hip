// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-HOSTCALL %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-BUFFERED %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcxx-exceptions -fexceptions -fclangir -fcuda-is-device -emit-llvm %s -o - | FileCheck --check-prefix=LLVM  %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=BUFFERED %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=BUFFERED %s

// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-HOSTCALL %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-BUFFERED %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=BUFFERED %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=BUFFERED %s

// printf fell through to a library call, which does not exist on the device:
// the reference survived to the device link and failed there as an undefined
// symbol. AMDGPU implements it through the OpenCL printf runtime.

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *, ...);

// CIR-HOSTCALL: module {{.*}} attributes {{.*}}cir.amdgpu_printf_kind = "hostcall"
// CIR-BUFFERED: module {{.*}} attributes {{.*}}cir.amdgpu_printf_kind = "buffered"

// CIR-LABEL: cir.func {{.*}} @_Z1pi(
// CIR: %[[FMT:.*]] = cir.cast array_to_ptrdecay
// CIR: %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @__cir_amdgpu_printf(%[[FMT]], %[[V]]) nothrow : (!cir.ptr<!s8i>, !s32i) -> !s32i
// CIR: cir.func private {{.*}} @__cir_amdgpu_printf(!cir.ptr<!s8i>, ...) -> !s32i
// LLVM-LABEL: @_Z1pi
// LLVM: call{{.*}} i64 @__ockl_printf_begin(i64
// LLVM: call{{.*}} i64 @__ockl_printf_append_string_n(i64
// LLVM: call{{.*}} i64 @__ockl_printf_append_args(i64
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z1pi
// BUFFERED: call{{.*}} ptr addrspace(1) @__printf_alloc(i32
// BUFFERED: end.block:
// BUFFERED: argpush.block:
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ void p(int v) { printf("%d\n", v); }

// The result of printf must be forwarded to its users. CIR places the
// expansion blocks after the block that uses the result, so the use is matched
// in any order.
// CIR-LABEL: cir.func {{.*}} @_Z1ri(
// CIR: %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR: %[[RES:.*]] = cir.call @__cir_amdgpu_printf({{.*}}) nothrow
// CIR: cir.store %[[RES]], %[[RETVAL]]
// LLVM-LABEL: @_Z1ri
// LLVM-DAG: [[RES:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64
// LLVM-DAG: [[RET:%.*]] = trunc i64 [[RES]] to i32
// LLVM-DAG: {{store|ret}} i32 [[RET]]
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z1ri
// BUFFERED-DAG: %printf_result = sext i1 %{{.*}} to i32
// BUFFERED-DAG: {{store|ret}} i32 %printf_result
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ int r(int v) { return printf("%d\n", v); }

// Device printf cannot throw, so a pending cleanup must not turn it into an
// invoke with an exception-handling path.
struct D { __device__ ~D(); };
// CIR-LABEL: cir.func {{.*}} @_Z2ehi(
// CIR: cir.cleanup.scope {
// CIR: cir.call @__cir_amdgpu_printf({{.*}}) nothrow
// CIR: } cleanup all {
// LLVM-LABEL: @_Z2ehi(
// LLVM-NOT: personality
// LLVM: call{{.*}} i64 @__ockl_printf_begin(i64
// LLVM-NOT: {{invoke|landingpad|resume}}
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z2ehi(
// BUFFERED-NOT: personality
// BUFFERED: call{{.*}} ptr addrspace(1) @__printf_alloc(i32
// BUFFERED-NOT: {{invoke|landingpad|resume}}
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ void eh(int v) { D d; printf("%d\n", v); }
