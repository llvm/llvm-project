#include "../Inputs/cuda.h"

// RUN: echo "sample fatbin" > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s


// CIR-HOST: module @"{{.*}}" attributes {
// CIR-HOST:   cir.cu.binary_handle = #cir.cu.binary_handle<{{.*}}.fatbin>,
// CIR-HOST:   cir.global_ctors = [#cir.global_ctor<"__cuda_module_ctor", {{[0-9]+}}>]
// CIR-HOST: }

// Module destructor goes here.
// This is not a real destructor, as explained in LoweringPrepare.

// CIR-HOST: cir.func internal private @__cuda_module_dtor() {
// CIR-HOST:   %[[#HandleGlobal:]] = cir.get_global @__cuda_gpubin_handle
// CIR-HOST:   %[[#Handle:]] = cir.load %0
// CIR-HOST:   cir.call @__cudaUnregisterFatBinary(%[[#Handle]])
// CIR-HOST: }

// CIR-HOST: cir.global "private" constant cir_private @".str_Z2fnv" =
// CIR-HOST-SAME: #cir.const_array<"_Z2fnv", trailing_zeros>

// COM: In OG this variable has an `unnamed_addr` attribute.
// LLVM-HOST: @.str_Z2fnv = private constant [7 x i8] c"_Z2fnv\00"

// The corresponding CIR test for these three variables are down below.
// They are here because LLVM IR puts global variables at the front of file.

// LLVM-HOST: @__cuda_fatbin_str = private constant [14 x i8] c"sample fatbin\0A", section ".nv_fatbin"
// LLVM-HOST: @__cuda_fatbin_wrapper = internal constant {
// LLVM-HOST:   i32 1180844977, i32 1, ptr @__cuda_fatbin_str, ptr null
// LLVM-HOST: }
// LLVM-HOST: @llvm.global_ctors = {{.*}}ptr @__cuda_module_ctor

// LLVM-HOST: define internal void @__cuda_module_dtor() {
// LLVM-HOST:   %[[#LLVMHandleVar:]] = load ptr, ptr @__cuda_gpubin_handle, align 8
// LLVM-HOST:   call void @__cudaUnregisterFatBinary(ptr %[[#LLVMHandleVar]])
// LLVM-HOST:   ret void
// LLVM-HOST: }

__global__ void fn() {}

__device__ int a;
__constant__ int b;

// CIR-HOST: cir.func internal private @__cuda_register_globals(%[[FatbinHandle:[a-zA-Z0-9]+]]{{.*}}) {
// CIR-HOST:   %[[#NULL:]] = cir.const #cir.ptr<null>
// CIR-HOST:   %[[#T1:]] = cir.get_global @".str_Z2fnv"
// CIR-HOST:   %[[#DeviceFn:]] = cir.cast bitcast %[[#T1]]
// CIR-HOST:   %[[#T2:]] = cir.get_global @_Z17__device_stub__fnv
// CIR-HOST:   %[[#HostFn:]] = cir.cast bitcast %[[#T2]]
// CIR-HOST:   %[[#MinusOne:]] = cir.const #cir.int<-1>
// CIR-HOST:   cir.call @__cudaRegisterFunction(
// CIR-HOST-SAME: %[[FatbinHandle]],
// CIR-HOST-SAME: %[[#HostFn]],
// CIR-HOST-SAME: %[[#DeviceFn]],
// CIR-HOST-SAME: %[[#DeviceFn]],
// CIR-HOST-SAME: %[[#MinusOne]],
// CIR-HOST-SAME: %[[#NULL]], %[[#NULL]], %[[#NULL]], %[[#NULL]], %[[#NULL]])
// Registration for __constant__ int b (isConstant=1):
// CIR-HOST:   %[[#T3:]] = cir.get_global @".strb0"
// CIR-HOST:   %[[#DeviceB:]] = cir.cast bitcast %[[#T3]]
// CIR-HOST:   %[[#T4:]] = cir.get_global @b
// CIR-HOST:   %[[#HostB:]] = cir.cast bitcast %[[#T4]]
// CIR-HOST:   %[[#ExtB:]] = cir.const #cir.int<0>
// CIR-HOST:   %[[#SzB:]] = cir.const #cir.int<4>
// CIR-HOST:   %[[#ConstB:]] = cir.const #cir.int<1>
// CIR-HOST:   %[[#ZeroB:]] = cir.const #cir.int<0>
// CIR-HOST:   cir.call @__cudaRegisterVar(%arg0, %[[#HostB]], %[[#DeviceB]], %[[#DeviceB]],
// CIR-HOST-SAME: %[[#ExtB]], %[[#SzB]], %[[#ConstB]], %[[#ZeroB]])
//
// Registration for __device__ int a (isConstant=0):
// CIR-HOST:   %[[#T5:]] = cir.get_global @".stra1"
// CIR-HOST:   %[[#DeviceA:]] = cir.cast bitcast %[[#T5]]
// CIR-HOST:   %[[#T6:]] = cir.get_global @a
// CIR-HOST:   %[[#HostA:]] = cir.cast bitcast %[[#T6]]
// CIR-HOST:   %[[#ExtA:]] = cir.const #cir.int<0>
// CIR-HOST:   %[[#SzA:]] = cir.const #cir.int<4>
// CIR-HOST:   %[[#ConstA:]] = cir.const #cir.int<0>
// CIR-HOST:   %[[#ZeroA:]] = cir.const #cir.int<0>
// CIR-HOST:   cir.call @__cudaRegisterVar(%arg0, %[[#HostA]], %[[#DeviceA]], %[[#DeviceA]],
// CIR-HOST-SAME: %[[#ExtA]], %[[#SzA]], %[[#ConstA]], %[[#ZeroA]])
// CIR-HOST: }

// LLVM-HOST: define internal void @__cuda_register_globals(ptr %[[#LLVMFatbin:]]) {
// LLVM-HOST:   call i32 @__cudaRegisterFunction(
// LLVM-HOST-SAME: ptr %[[#LLVMFatbin]],
// LLVM-HOST-SAME: ptr @_Z17__device_stub__fnv,
// LLVM-HOST-SAME: ptr @.str_Z2fnv,
// LLVM-HOST-SAME: ptr @.str_Z2fnv,
// LLVM-HOST-SAME: i32 -1,
// LLVM-HOST-SAME: ptr null, ptr null, ptr null, ptr null, ptr null)
// LLVM-HOST:   call void @__cudaRegisterVar(
// LLVM-HOST-SAME: ptr %0, ptr @b, ptr @.strb0, ptr @.strb0,
// LLVM-HOST-SAME: i32 0, i64 4, i32 1, i32 0)
// LLVM-HOST:   call void @__cudaRegisterVar(
// LLVM-HOST-SAME: ptr %0, ptr @a, ptr @.stra1, ptr @.stra1,
// LLVM-HOST-SAME: i32 0, i64 4, i32 0, i32 0)
// LLVM-HOST: }

// The content in const array should be the same as echoed above,
// with a trailing line break ('\n', 0x0A).
// CIR-HOST: cir.global "private" constant cir_private @__cuda_fatbin_str =
// CIR-HOST-SAME: #cir.const_array<"sample fatbin\0A">
// CIR-HOST-SAME: {{.*}}section = ".nv_fatbin"

// The first value is CUDA file head magic number.
// CIR-HOST: cir.global "private" constant internal @__cuda_fatbin_wrapper
// CIR-HOST: = #cir.const_record<{
// CIR-HOST:   #cir.int<1180844977> : !s32i,
// CIR-HOST:   #cir.int<1> : !s32i,
// CIR-HOST:   #cir.global_view<@__cuda_fatbin_str> : !cir.ptr<!void>,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>
// CIR-HOST: }>
// CIR-HOST-SAME: {{.*}}section = ".nvFatBinSegment"

// CIR-HOST: cir.func {{.*}} @__cudaRegisterFatBinary
// CIR-HOST: cir.func {{.*}} @__cuda_module_ctor() {
// CIR-HOST:   %[[#Fatbin:]] = cir.call @__cudaRegisterFatBinary
// CIR-HOST:   %[[#FatbinGlobal:]] = cir.get_global @__cuda_gpubin_handle
// CIR-HOST:   cir.store %[[#Fatbin]], %[[#FatbinGlobal]]
// CIR-HOST:   cir.call @__cuda_register_globals
// CIR-HOST:   cir.call @__cudaRegisterFatBinaryEnd
// CIR-HOST:   %[[#ModuleDtor:]] = cir.get_global @__cuda_module_dtor
// CIR-HOST:   cir.call @atexit(%[[#ModuleDtor]])
// CIR-HOST: }

// LLVM-HOST: define internal void @__cuda_module_ctor() {
// LLVM-HOST:   %[[#LLVMFatbin:]] = call ptr @__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// LLVM-HOST:   store ptr %[[#LLVMFatbin]], ptr @__cuda_gpubin_handle
// LLVM-HOST:   call void @__cuda_register_globals
// LLVM-HOST:   call void @__cudaRegisterFatBinaryEnd
// LLVM-HOST:   call i32 @atexit(ptr @__cuda_module_dtor)
// LLVM-HOST: }

// OGCG-HOST: @a = internal global i32 undef, align 4
// OGCG-HOST: @b = internal global i32 undef, align 4
// OGCG-HOST: @0 = private unnamed_addr constant [7 x i8] c"_Z2fnv\00", align 1
// OGCG-HOST: @1 = private unnamed_addr constant [2 x i8] c"a\00", align 1
// OGCG-HOST: @2 = private unnamed_addr constant [2 x i8] c"b\00", align 1
// OGCG-HOST: @3 = private constant [14 x i8] c"sample fatbin\0A", section ".nv_fatbin", align 8
// OGCG-HOST: @__cuda_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @3, ptr null }, section ".nvFatBinSegment", align 8
// OGCG-HOST: @__cuda_gpubin_handle = internal global ptr null, align 8
// OGCG-HOST: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cuda_module_ctor, ptr null }]

// OGCG-HOST: define internal void @__cuda_register_globals(ptr %[[#HANDLE:]]) {
// OGCG-HOST: entry:
// OGCG-HOST:   %1 = call i32 @__cudaRegisterFunction(ptr %[[#HANDLE]],
// OGCG-HOST-SAME: ptr @_Z17__device_stub__fnv,
// OGCG-HOST-SAME: ptr @0,
// OGCG-HOST-SAME: ptr @0,
// OGCG-HOST-SAME: i32 -1,
// OGCG-HOST-SAME: ptr null,
// OGCG-HOST-SAME: ptr null,
// OGCG-HOST-SAME: ptr null,
// OGCG-HOST-SAME: ptr null,
// OGCG-HOST-SAME: ptr null)
// OGCG-HOST:   call void @__cudaRegisterVar(ptr %[[#HANDLE]],
// OGCG-HOST-SAME: ptr @a,
// OGCG-HOST-SAME: ptr @1,
// OGCG-HOST-SAME: ptr @1,
// OGCG-HOST-SAME: i32 0,
// OGCG-HOST-SAME: i64 4,
// OGCG-HOST-SAME: i32 0,
// OGCG-HOST-SAME: i32 0)
// OGCG-HOST:   call void @__cudaRegisterVar(ptr %[[#HANDLE]],
// OGCG-HOST-SAME: ptr @b,
// OGCG-HOST-SAME: ptr @2,
// OGCG-HOST-SAME: ptr @2,
// OGCG-HOST-SAME: i32 0,
// OGCG-HOST-SAME: i64 4,
// OGCG-HOST-SAME: i32 1,
// OGCG-HOST-SAME: i32 0)
// OGCG-HOST:   ret void
// OGCG-HOST: }

// OGCG-HOST: define internal void @__cuda_module_ctor() {
// OGCG-HOST: entry:
// OGCG-HOST:   %[[#WRAPADDR:]] = call ptr @__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// OGCG-HOST:   store ptr %[[#WRAPADDR]], ptr @__cuda_gpubin_handle, align 8
// OGCG-HOST:   call void @__cuda_register_globals(ptr %[[#WRAPADDR]])
// OGCG-HOST:   call void @__cudaRegisterFatBinaryEnd(ptr %[[#WRAPADDR]])
// OGCG-HOST:   %1 = call i32 @atexit(ptr @__cuda_module_dtor)
// OGCG-HOST:   ret void
// OGCG-HOST: }

// OGCG-HOST: define internal void @__cuda_module_dtor() {
// OGCG-HOST: entry:
// OGCG-HOST:   %[[#HANDLE:]] = load ptr, ptr @__cuda_gpubin_handle, align 8
// OGCG-HOST:   call void @__cudaUnregisterFatBinary(ptr %[[#HANDLE]])
// OGCG-HOST:   ret void
// OGCG-HOST: }
