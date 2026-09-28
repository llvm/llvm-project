// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcxx-exceptions -fexceptions \
// RUN: -fclangir -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcxx-exceptions -fexceptions \
// RUN: -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcxx-exceptions -fexceptions \
// RUN: -fclangir -fcuda-is-device -mprintf-kind=buffered -emit-llvm %s -o %t-cir-buffered.ll
// RUN: FileCheck --check-prefix=BUFFERED --input-file=%t-cir-buffered.ll %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -std=c++11 \
// RUN: -fcxx-exceptions -fexceptions \
// RUN: -fcuda-is-device -mprintf-kind=buffered -emit-llvm %s -o %t-buffered.ll
// RUN: FileCheck --check-prefix=BUFFERED --input-file=%t-buffered.ll %s

// printf fell through to a library call, which does not exist on the device:
// the reference survived to the device link and failed there as an undefined
// symbol. AMDGPU implements it through the OpenCL printf runtime.

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *, ...);

// LLVM-LABEL: @_Z1pi
// LLVM: call i64 @__ockl_printf_begin(i64
// LLVM: call i64 @__ockl_printf_append_string_n(i64
// LLVM: call i64 @__ockl_printf_append_args(i64
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z1pi
// BUFFERED: call ptr addrspace(1) @__printf_alloc(i32
// BUFFERED: end.block:
// BUFFERED: argpush.block:
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ void p(int v) { printf("%d\n", v); }

// The result of printf must be forwarded to its users. CIR places the
// expansion blocks after the block that uses the result, so the use is matched
// in any order.
// LLVM-LABEL: @_Z1ri
// LLVM-DAG: [[RES:%.*]] = call i64 @__ockl_printf_append_args(i64
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
// LLVM-LABEL: @_Z2ehi(
// LLVM-NOT: personality
// LLVM: call i64 @__ockl_printf_begin(i64
// LLVM-NOT: {{invoke|landingpad|resume}}
// LLVM-NOT: call{{.*}}@__cir_amdgpu_printf
// BUFFERED-LABEL: @_Z2ehi(
// BUFFERED-NOT: personality
// BUFFERED: call ptr addrspace(1) @__printf_alloc(i32
// BUFFERED-NOT: {{invoke|landingpad|resume}}
// BUFFERED-NOT: call{{.*}}@__cir_amdgpu_printf
__device__ void eh(int v) { D d; printf("%d\n", v); }
