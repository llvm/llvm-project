#include "../Inputs/cuda.h"

// RUN: echo "sample fatbin" > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-cir -fhip-new-launch-api -I%S/../Inputs/ \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-llvm -fhip-new-launch-api  -I%S/../Inputs/ \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// OGCG emits LLVM IR in different order than clangir, we add at the end the order of OGCG.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  \
// RUN:            -x hip -emit-llvm -fhip-new-launch-api  -I%S/../Inputs/ \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s


// CIR-HOST: module @"{{.*}}" attributes {
// CIR-HOST:   cir.cu.binary_handle = #cir.cu.binary_handle<{{.*}}.fatbin>,
// CIR-HOST:   cir.global_ctors = [#cir.global_ctor<"__hip_module_ctor", {{[0-9]+}}>]
// CIR-HOST: }

// LLVM-HOST: @.strb0 = private constant [2 x i8] c"b\00"
// LLVM-HOST: @.stra1 = private constant [2 x i8] c"a\00"
// LLVM-HOST: @.str_Z2fnv = private constant [7 x i8] c"_Z2fnv\00"
// LLVM-HOST: @__hip_fatbin_str = private constant [14 x i8] c"sample fatbin\0A", section ".hip_fatbin"
// LLVM-HOST: @__hip_fatbin_wrapper = internal constant {
// LLVM-HOST:   i32 1212764230, i32 1, ptr @__hip_fatbin_str, ptr null
// LLVM-HOST: }, section ".hipFatBinSegment"
// LLVM-HOST: @_Z2fnv = constant ptr @_Z17__device_stub__fnv, align 8
// LLVM-HOST: @a = internal global i32 undef, align 4
// LLVM-HOST: @b = internal global i32 undef, align 4
// LLVM-HOST: @llvm.global_ctors = {{.*}}ptr @__hip_module_ctor

// CIR-HOST:  cir.func internal private @__hip_module_dtor() {
// CIR-HOST:   %[[#HandleGlobal:]] = cir.get_global @__hip_gpubin_handle
// CIR-HOST:   %[[#HandleAddr:]] = cir.load %[[#HandleGlobal]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   %[[#NullVal:]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   %3 = cir.cmp(ne, %[[#HandleAddr]], %[[#NullVal]]) : !cir.ptr<!cir.ptr<!void>>, !cir.bool loc(#loc)
// CIR-HOST:    cir.brcond %3 ^bb1, ^bb2 loc(#loc)
// CIR-HOST:  ^bb1:
// CIR-HOST:    cir.call @__hipUnregisterFatBinary(%[[#HandleAddr]]) : (!cir.ptr<!cir.ptr<!void>>) -> () loc(#loc)
// CIR-HOST:    %[[#HandleAddr:]] = cir.get_global @__hip_gpubin_handle : !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:    cir.store %[[#NullVal]], %[[#HandleAddr]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:    cir.br ^bb2 loc(#loc)
// CIR-HOST:  ^bb2:  // 2 preds: ^bb0, ^bb1
// CIR-HOST:    cir.return loc(#loc)
// CIR-HOST:  } loc(#loc)

// LLVM-HOST: define internal void @__hip_module_dtor() {
// LLVM-HOST:    %[[#LLVMHandleVar:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// LLVM-HOST:    %[[#ICMP:]] = icmp ne ptr %[[#LLVMHandleVar]], null
// LLVM-HOST:    br i1 %[[#ICMP]], label %[[IFBLOCK:[^,]+]], label %[[EXITBLOCK:[^,]+]]
// LLVM-HOST:  [[IFBLOCK]]:                                               ; preds = %0
// LLVM-HOST:    call void @__hipUnregisterFatBinary(ptr %[[#LLVMHandleVar]])
// LLVM-HOST:    store ptr null, ptr @__hip_gpubin_handle, align 8
// LLVM-HOST:    br label %[[EXITBLOCK]]
// LLVM-HOST:  [[EXITBLOCK]]:                                             ; preds = %[[IFBLOCK]], %0
// LLVM-HOST:    ret void
// LLVM-HOST:  }

// CIR-HOST: cir.global "private" constant cir_private @".str_Z2fnv" =
// CIR-HOST-SAME: #cir.const_array<"_Z2fnv", trailing_zeros>
 
__global__ void fn() {}


__device__ int a;
__constant__ int b;

// CIR-HOST: cir.func internal private @__hip_register_globals(%[[FatbinHandle:[a-zA-Z0-9]+]]{{.*}}) {
// CIR-HOST:   %[[#NULL:]] = cir.const #cir.ptr<null>
// CIR-HOST:   %[[#T1:]] = cir.get_global @".str_Z2fnv"
// CIR-HOST:   %[[#DeviceFn:]] = cir.cast bitcast %[[#T1]]
// CIR-HOST:   %[[#T2:]] = cir.get_global @_Z2fnv
// CIR-HOST:   %[[#HostFnHandle:]] = cir.cast bitcast %[[#T2]]
// CIR-HOST:   %[[#MinusOne:]] = cir.const #cir.int<-1>
// CIR-HOST:   cir.call @__hipRegisterFunction(
// CIR-HOST-SAME: %[[FatbinHandle]],
// CIR-HOST-SAME: %[[#HostFnHandle]],
// CIR-HOST-SAME: %[[#DeviceFn]],
// CIR-HOST-SAME: %[[#DeviceFn]],
// CIR-HOST-SAME: %[[#MinusOne]],
// CIR-HOST-SAME: %[[#NULL]], %[[#NULL]], %[[#NULL]], %[[#NULL]], %[[#NULL]])
// Registration for __constant__ int b (isConstant=1):
// CIR-HOST: %[[#T3:]] = cir.get_global @".strb0"
// CIR-HOST: %[[#DeviceB:]] = cir.cast bitcast %[[#T3]]
// CIR-HOST: %[[#T4:]] = cir.get_global @b
// CIR-HOST: %[[#HostB:]] = cir.cast bitcast %[[#T4]]
// CIR-HOST: %[[#ExtB:]] = cir.const #cir.int<0>
// CIR-HOST: %[[#SzB:]] = cir.const #cir.int<4>
// CIR-HOST: %[[#ConstB:]] = cir.const #cir.int<1>
// CIR-HOST: %[[#ZeroB:]] = cir.const #cir.int<0>
// CIR-HOST: cir.call @__hipRegisterVar(%[[FatbinHandle]],
// CIR-HOST-SAME: %[[#HostB]],
// CIR-HOST-SAME: %[[#DeviceB]],
// CIR-HOST-SAME: %[[#DeviceB]],
// CIR-HOST-SAME: %[[#ExtB]],
// CIR-HOST-SAME: %[[#SzB]],
// CIR-HOST-SAME: %[[#ConstB]],
// CIR-HOST-SAME: %[[#ZeroB]])
//
// Registration for __device__ int a (isConstant=0):
// CIR-HOST: %[[#T5:]] = cir.get_global @".stra1"
// CIR-HOST: %[[#DeviceA:]] = cir.cast bitcast %[[#T5]]
// CIR-HOST: %[[#T6:]] = cir.get_global @a
// CIR-HOST: %[[#HostA:]] = cir.cast bitcast %[[#T6]]
// CIR-HOST: %[[#ExtA:]] = cir.const #cir.int<0>
// CIR-HOST: %[[#SzA:]] = cir.const #cir.int<4>
// CIR-HOST: %[[#ConstA:]] = cir.const #cir.int<0>
// CIR-HOST: %[[#ZeroA:]] = cir.const #cir.int<0>
// CIR-HOST: cir.call @__hipRegisterVar(%[[FatbinHandle]],
// CIR-HOST-SAME: %[[#HostA]],
// CIR-HOST-SAME: %[[#DeviceA]],
// CIR-HOST-SAME: %[[#DeviceA]],
// CIR-HOST-SAME: %[[#ExtA]],
// CIR-HOST-SAME: %[[#SzA]],
// CIR-HOST-SAME: %[[#ConstA]],
// CIR-HOST-SAME: %[[#ZeroA]])
// CIR-HOST: cir.return loc(#loc)
// CIR-HOST: }

// LLVM-HOST: define internal void @__hip_register_globals(ptr %[[#LLVMFatbin:]]) {
// LLVM-HOST:   call i32 @__hipRegisterFunction(
// LLVM-HOST-SAME: ptr %[[#LLVMFatbin]],
// LLVM-HOST-SAME: ptr @_Z2fnv,
// LLVM-HOST-SAME: ptr @.str_Z2fnv,
// LLVM-HOST-SAME: ptr @.str_Z2fnv,
// LLVM-HOST-SAME: i32 -1,
// LLVM-HOST-SAME: ptr null, ptr null, ptr null, ptr null, ptr null)
// LLVM-HOST:   call void @__hipRegisterVar(
// LLVM-HOST-SAME: ptr %0, ptr @b, ptr @.strb0, ptr @.strb0,
// LLVM-HOST-SAME: i32 0, i64 4, i32 1, i32 0)
// LLVM-HOST:   call void @__hipRegisterVar(
// LLVM-HOST-SAME: ptr %0, ptr @a, ptr @.stra1, ptr @.stra1,
// LLVM-HOST-SAME: i32 0, i64 4, i32 0, i32 0)
// LLVM-HOST: }

// The content in const array should be the same as echoed above,
// with a trailing line break ('\n', 0x0A).
// CIR-HOST: cir.global "private" constant cir_private @__hip_fatbin_str =
// CIR-HOST-SAME: #cir.const_array<"sample fatbin\0A">
// CIR-HOST-SAME: {{.*}}section = ".hip_fatbin"

// The first value is HIP file head magic number.
// CIR-HOST: cir.global "private" constant internal @__hip_fatbin_wrapper
// CIR-HOST: = #cir.const_record<{
// CIR-HOST:   #cir.int<1212764230> : !s32i,
// CIR-HOST:   #cir.int<1> : !s32i,
// CIR-HOST:   #cir.global_view<@__hip_fatbin_str> : !cir.ptr<!void>,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>
// CIR-HOST: }>
// CIR-HOST-SAME: {{.*}}section = ".hipFatBinSegment"

// CIR-HOST: cir.func internal private @__hip_module_ctor() {
// CIR-HOST:   %[[#HandleGlobalVar:]] = cir.get_global @__hip_gpubin_handle : !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:   %[[#HandleAddr:]] = cir.load %[[#HandleGlobalVar]] : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   %[[#NullVal:]] = cir.const #cir.ptr<null> : !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   %[[#ICMP:]] = cir.cmp(eq, %[[#HandleAddr]], %[[#NullVal]]) : !cir.ptr<!cir.ptr<!void>>, !cir.bool loc(#loc)
// CIR-HOST:   cir.brcond %[[#ICMP]] ^bb1, ^bb2 loc(#loc)
// CIR-HOST: ^bb1:
// CIR-HOST:   %[[#FatBinWrapper:]] = cir.get_global @__hip_fatbin_wrapper : !cir.ptr<!rec_anon_struct> loc(#loc)
// CIR-HOST:   %[[#CastGlobalFatBin:]] = cir.cast bitcast %[[#FatBinWrapper]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!void> loc(#loc)
// CIR-HOST:   %[[#RTVal:]] = cir.call @__hipRegisterFatBinary(%[[#CastGlobalFatBin]]) : (!cir.ptr<!void>) -> !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   %[[#HandleGlobalVar:]] = cir.get_global @__hip_gpubin_handle : !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:   cir.store %[[#RTVal]], %[[#HandleGlobalVar]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:   cir.br ^bb2 loc(#loc)
// CIR-HOST: ^bb2:
// CIR-HOST:   %[[#HandleGlobalVar:]] = cir.get_global @__hip_gpubin_handle : !cir.ptr<!cir.ptr<!cir.ptr<!void>>> loc(#loc)
// CIR-HOST:   %[[#HandleVal:]] = cir.load %8 : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>, !cir.ptr<!cir.ptr<!void>> loc(#loc)
// CIR-HOST:   cir.call @__hip_register_globals(%[[#HandleVal]]) : (!cir.ptr<!cir.ptr<!void>>) -> () loc(#loc)
// CIR-HOST:   %[[#DTOR:]] = cir.get_global @__hip_module_dtor : !cir.ptr<!cir.func<()>> loc(#loc)
// CIR-HOST:   %11 = cir.call @atexit(%[[#DTOR]]) : (!cir.ptr<!cir.func<()>>) -> !s32i loc(#loc)
// CIR-HOST:   cir.return loc(#loc)
// CIR-HOST: } loc(#loc)

// LLVM-HOST: define internal void @__hip_module_ctor() {
// LLVM-HOST:  %[[#LLVMHandleVar:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// LLVM-HOST:  %[[#ICMP:]] = icmp eq ptr %[[#LLVMHandleVar]], null
// LLVM-HOST:  br i1 %[[#ICMP]], label %[[IFBLOCK:[^,]+]], label %[[EXITBLOCK:[^,]+]]
// LLVM-HOST: [[IFBLOCK]]:
// LLVM-HOST:  %[[#Value:]] = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
// LLVM-HOST:  store ptr %[[#Value]], ptr @__hip_gpubin_handle, align 8
// LLVM-HOST:  br label %[[EXITBLOCK]]
// LLVM-HOST: [[EXITBLOCK]]:
// LLVM-HOST:  %[[#HandleValue:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// LLVM-HOST:  call void @__hip_register_globals(ptr %[[#HandleValue]])
// LLVM-HOST:  call i32 @atexit(ptr @__hip_module_dtor)
// LLVM-HOST:  ret void

// OGCG-HOST: @_Z2fnv = constant ptr @_Z17__device_stub__fnv, align 8
// OGCG-HOST: @a = internal global i32 undef, align 4
// OGCG-HOST: @b = internal global i32 undef, align 4
// OGCG-HOST: @0 = private unnamed_addr constant [7 x i8] c"_Z2fnv\00", align 1
// OGCG-HOST: @1 = private unnamed_addr constant [2 x i8] c"a\00", align 1
// OGCG-HOST: @2 = private unnamed_addr constant [2 x i8] c"b\00", align 1
// OGCG-HOST: @3 = private constant [14 x i8] c"sample fatbin\0A", section ".hip_fatbin", align 4096
// OGCG-HOST: @__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @3, ptr null }, section ".hipFatBinSegment", align 8
// OGCG-HOST: @__hip_gpubin_handle = internal global ptr null, align 8
// OGCG-HOST: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }]

// OGCG-HOST: define internal void @__hip_register_globals(ptr %[[#LLVMFatbin:]]) {
// OGCG-HOST: entry:
// OGCG-HOST:   call i32 @__hipRegisterFunction(
// OGCG-HOST-SAME: ptr %[[#LLVMFatbin]],
// OGCG-HOST-SAME: ptr @_Z2fnv,
// OGCG-HOST-SAME: ptr @0,
// OGCG-HOST-SAME: ptr @0,
// OGCG-HOST-SAME: i32 -1,
// OGCG-HOST-SAME: ptr null, ptr null, ptr null, ptr null, ptr null)
// OGCG-HOST:   call void @__hipRegisterVar(
// OGCG-HOST-SAME: ptr %[[#LLVMFatbin]],
// OGCG-HOST-SAME: ptr @a,
// OGCG-HOST-SAME: ptr @1,
// OGCG-HOST-SAME: ptr @1,
// OGCG-HOST-SAME: i32 0,
// OGCG-HOST-SAME: i64 4,
// OGCG-HOST-SAME: i32 0, i32 0)
// OGCG-HOST:   call void @__hipRegisterVar(
// OGCG-HOST-SAME: ptr %[[#LLVMFatbin]],
// OGCG-HOST-SAME: ptr @b,
// OGCG-HOST-SAME: ptr @2,
// OGCG-HOST-SAME: ptr @2,
// OGCG-HOST-SAME: i32 0,
// OGCG-HOST-SAME: i64 4,
// OGCG-HOST-SAME: i32 1, i32 0)
// OGCG-HOST: ret void
// OGCG-HOST: }

// OGCG-HOST: define internal void @__hip_module_ctor() {
// OGCG-HOST:  %[[#LLVMHandleVar:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// OGCG-HOST:  %[[#ICMP:]] = icmp eq ptr %[[#LLVMHandleVar]], null
// OGCG-HOST:  br i1 %[[#ICMP]], label %[[IFBLOCK:[^,]+]], label %[[EXITBLOCK:[^,]+]]
// OGCG-HOST: [[IFBLOCK]]:
// OGCG-HOST:  %[[#Value:]] = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
// OGCG-HOST:  store ptr %[[#Value]], ptr @__hip_gpubin_handle, align 8
// OGCG-HOST:  br label %[[EXITBLOCK]]
// OGCG-HOST: [[EXITBLOCK]]:
// OGCG-HOST:  %[[#HandleValue:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// OGCG-HOST:  call void @__hip_register_globals(ptr %[[#HandleValue]])
// OGCG-HOST:  call i32 @atexit(ptr @__hip_module_dtor)
// OGCG-HOST:  ret void

// OGCG-HOST: define internal void @__hip_module_dtor() {
// OGCG-HOST:  entry:
// OGCG-HOST:    %[[#LLVMHandleVar:]] = load ptr, ptr @__hip_gpubin_handle, align 8
// OGCG-HOST:    %[[#ICMP:]] = icmp ne ptr %[[#LLVMHandleVar]], null
// OGCG-HOST:    br i1 %[[#ICMP]], label %[[IFBLOCK:[^,]+]], label %[[EXITBLOCK:[^,]+]]
// OGCG-HOST:  [[IFBLOCK]]:                                               ; preds = %entry
// OGCG-HOST:    call void @__hipUnregisterFatBinary(ptr %[[#LLVMHandleVar]])
// OGCG-HOST:    store ptr null, ptr @__hip_gpubin_handle, align 8
// OGCG-HOST:    br label %[[EXITBLOCK]]
// OGCG-HOST:  [[EXITBLOCK]]:                                             ; preds = %[[IFBLOCK]], %entry
// OGCG-HOST:    ret void
// OGCG-HOST:  }

