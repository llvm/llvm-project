#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefixes=CIR,CIR-HIP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVM-HIP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVM-HIP

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -target-sdk-version=12.0 -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefixes=CIR,CIR-CUDA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -target-sdk-version=12.0 -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVM-CUDA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -target-sdk-version=12.0 -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVM-CUDA

// On the host, the address of a kernel is the pointer the offload runtime
// registers for it. For HIP that is the kernel handle, a global holding the
// address of the device stub, so launching through a kernel pointer loads the
// stub from the handle. For CUDA it is the device stub itself.

__global__ void kern(int *p) {}
template <class T> __global__ void tkern(T *p) {}

const void *table[] = {(const void *)kern, (const void *)tkern<float>};

// CIR-HIP:  cir.global external @table = #cir.const_array<[#cir.global_view<@_Z4kernPi> : !cir.ptr<!void>, #cir.global_view<@_Z5tkernIfEvPT_> : !cir.ptr<!void>]>
// CIR-CUDA: cir.global external @table = #cir.const_array<[#cir.global_view<@_Z19__device_stub__kernPi> : !cir.ptr<!void>, #cir.global_view<@_Z20__device_stub__tkernIfEvPT_> : !cir.ptr<!void>]>

// LLVM-HIP:  @table = global [2 x ptr] [ptr @_Z4kernPi, ptr @_Z5tkernIfEvPT_]
// LLVM-CUDA: @table = global [2 x ptr] [ptr @_Z19__device_stub__kernPi, ptr @_Z20__device_stub__tkernIfEvPT_]

const void *addr() { return (const void *)kern; }

// CIR-LABEL: cir.func {{.*}} @_Z4addrv(
// CIR-HIP:     %[[HANDLE:.+]] = cir.get_global @_Z4kernPi : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>>
// CIR-HIP:     cir.cast bitcast %[[HANDLE]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>> -> !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>
// CIR-CUDA:    cir.get_global @_Z19__device_stub__kernPi : !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>

// LLVM-LABEL: define {{.*}} ptr @_Z4addrv(
// LLVM-HIP:     {{(ret|store)}} ptr @_Z4kernPi
// LLVM-CUDA:    {{(ret|store)}} ptr @_Z19__device_stub__kernPi

const void *tmpl_addr() { return (const void *)tkern<float>; }

// CIR-LABEL: cir.func {{.*}} @_Z9tmpl_addrv(
// CIR-HIP:     cir.get_global @_Z5tkernIfEvPT_ : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!cir.float>)>>>
// CIR-CUDA:    cir.get_global @_Z20__device_stub__tkernIfEvPT_ : !cir.ptr<!cir.func<(!cir.ptr<!cir.float>)>>

// LLVM-LABEL: define {{.*}} ptr @_Z9tmpl_addrv(
// LLVM-HIP:     {{(ret|store)}} ptr @_Z5tkernIfEvPT_
// LLVM-CUDA:    {{(ret|store)}} ptr @_Z20__device_stub__tkernIfEvPT_

void indirect(void (*f)(int *), int *p) { f<<<1, 1>>>(p); }

// CIR-LABEL: cir.func {{.*}} @_Z8indirectPFvPiES_(
// CIR:         %[[F:.+]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>
// CIR-HIP:     %[[HANDLE:.+]] = cir.cast bitcast %[[F]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>> -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>>
// CIR-HIP:     %[[STUB:.+]] = cir.load {{.*}} %[[HANDLE]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>
// CIR-HIP:     cir.call %[[STUB]](
// CIR-CUDA:    cir.call %[[F]](

// LLVM-LABEL: define {{.*}} void @_Z8indirectPFvPiES_(
// LLVM:         [[F:%.+]] = load ptr, ptr %{{.+}}, align 8
// LLVM-HIP:     [[STUB:%.+]] = load ptr, ptr [[F]], align 8
// LLVM-HIP:     call void [[STUB]](
// LLVM-CUDA:    call void [[F]](

void direct(int *p) { kern<<<1, 1>>>(p); }

// CIR-LABEL: cir.func {{.*}} @_Z6directPi(
// CIR:         cir.call @_Z19__device_stub__kernPi(

// LLVM-LABEL: define {{.*}} void @_Z6directPi(
// LLVM:         call void @_Z19__device_stub__kernPi(
