// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=8.0 -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:       --check-prefixes=ALL,LNX,NORDC,CUDA,CUDANORDC,CUDA-OLD
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=8.0  -fcuda-include-gpubinary %t \
// RUN:     -o - -DNOGLOBALS \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:     -check-prefixes=NOGLOBALS,CUDANOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=8.0 -fgpu-rdc -fcuda-include-gpubinary %t \
// RUN:     -o - \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:       --check-prefixes=ALL,LNX,RDC,CUDA,CUDARDC,CUDA-OLD
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=8.0 -o - \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=NOGPUBIN

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s       \
// RUN:     -target-sdk-version=9.2 -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s -allow-deprecated-dag-overlap \
// RUN:       --check-prefixes=ALL,LNX,NORDC,CUDA,CUDANORDC,CUDA-NEW
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=9.2 -fcuda-include-gpubinary %t -o -  -DNOGLOBALS \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:       --check-prefixes=NOGLOBALS,CUDANOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=9.2 -fgpu-rdc -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s -allow-deprecated-dag-overlap \
// RUN:       --check-prefixes=ALL,LNX,RDC,CUDA,CUDARDC,CUDA-NEW
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -std=c++17 \
// RUN:     -target-sdk-version=9.2 -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s -allow-deprecated-dag-overlap \
// RUN:       --check-prefixes=ALL,LNX,NORDC,CUDA,CUDANORDC,CUDA-NEW,LNX_17,NORDC17
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -std=c++17 \
// RUN:     -target-sdk-version=9.2 -fgpu-rdc -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s -allow-deprecated-dag-overlap \
// RUN:       --check-prefixes=ALL,LNX,RDC,CUDA,CUDARDC,CUDA-NEW,LNX_17,RDC17
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -target-sdk-version=9.2 -o - \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=NOGPUBIN

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=ALL,LNX,NORDC,HIP,HIPEF
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o -  -DNOGLOBALS -x hip \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=NOGLOBALS,HIPNOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fgpu-rdc -fcuda-include-gpubinary %t -o - -x hip \
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=ALL,LNX,RDC,HIP,HIPEF
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=ALL,LNX,NORDC,HIP,HIPNEF

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -aux-triple amdgcn -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=ALL,WIN

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -aux-triple amdgcn -emit-llvm %s \
// RUN:     -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=ALL,WIN,HIP,HIPNEF

#include "Inputs/cuda.h"

// HIPNEF: $__hip_gpubin_handle = comdat any

#ifndef NOGLOBALS
// NORDC-DAG: @device_var = internal global i32
// RDC-DAG: @device_var = global i32
// WIN-DAG: @"?device_var@@3HA" = internal global i32
__device__ int device_var;

// NORDC-DAG: @constant_var = internal global i32
// RDC-DAG: @constant_var = global i32
// WIN-DAG: @"?constant_var@@3HA" = internal global i32
__constant__ int constant_var;

// NORDC-DAG: @shared_var = internal global i32
// RDC-DAG: @shared_var = global i32
// WIN-DAG: @"?shared_var@@3HA" = internal global i32
__shared__ int shared_var;

// Make sure host globals don't get internalized...
// LNX-DAG: @host_var ={{.*}} global i32
// WIN-DAG: @"?host_var@@3HA" = dso_local global i32
int host_var;
// ... and that extern vars remain external.
// LNX-DAG: @ext_host_var = external global i32
// WIN-DAG: @"?ext_host_var@@3HA" = external dso_local global i32
extern int ext_host_var;

// external device-side variables -> extern references to their shadows.
// LNX-DAG: @ext_device_var = external global i32
// WIN-DAG: @"?ext_device_var@@3HA" = external dso_local global i32
extern __device__ int ext_device_var;
// LNX-DAG: @ext_device_var = external global i32
// WIN-DAG: @"?ext_constant_var@@3HA" = external dso_local global i32
extern __constant__ int ext_constant_var;

// external device-side variables with definitions should generate
// definitions for the shadows.
// NORDC-DAG: @ext_device_var_def = internal global i32 undef,
// RDC-DAG: @ext_device_var_def = global i32 undef,
// WIN-DAG: @"?ext_device_var_def@@3HA" = internal global i32 undef
extern __device__ int ext_device_var_def;
__device__ int ext_device_var_def = 1;
// NORDC-DAG: @ext_device_var_def = internal global i32 undef,
// RDC-DAG: @ext_device_var_def = global i32 undef,
// WIN-DAG: @"?ext_constant_var_def@@3HA" = internal global i32 undef
__constant__ int ext_constant_var_def = 2;

#if __cplusplus > 201402L
// NORDC17: @inline_var = internal global i32 undef, comdat, align 4{{$}}
// RDC17: @inline_var = linkonce_odr global i32 undef, comdat, align 4{{$}}
// NORDC17-NOT: @inline_var2 =
// RDC17-NOT: @inline_var2 =
// NORDC17: @_ZN1C17member_inline_varE = internal constant i32 undef, comdat, align 4{{$}}
// RDC17: @_ZN1C17member_inline_varE = linkonce_odr constant i32 undef, comdat, align 4{{$}}
// Check inline variable ODR-used by host is emitted on host and registered.
__device__ inline int inline_var = 3;
// Check inline variable not ODR-used by host is not emitted on host or registered.
__device__ inline int inline_var2 = 5;
struct C {
  __device__ static constexpr int member_inline_var = 4;
};
#endif

void use_pointers() {
  const int *p;
  p = &device_var;
  p = &constant_var;
  p = &shared_var;
  p = &host_var;
  p = &ext_device_var;
  p = &ext_constant_var;
  p = &ext_host_var;
#if __cplusplus > 201402L
  p = &inline_var;
  decltype(inline_var2) tmp;
  p = &C::member_inline_var;
#endif
}

__device__ void device_use() {
#if __cplusplus > 201402L
  const int *p = &inline_var2;
#endif
}

// Make sure that all parts of GPU code init/cleanup are there:
// * constant unnamed string with the device-side kernel name to be passed to
//   __hipRegisterFunction/__cudaRegisterFunction.
// ALL: @0 = private unnamed_addr constant [18 x i8] c"_Z10kernelfunciii\00"
// * constant unnamed string with the device-side kernel name to be passed to
//   __hipRegisterVar/__cudaRegisterVar.
// ALL: @1 = private unnamed_addr constant [11 x i8] c"device_var\00"
// ALL: @2 = private unnamed_addr constant [13 x i8] c"constant_var\00"
// ALL: @3 = private unnamed_addr constant [19 x i8] c"ext_device_var_def\00"
// ALL: @4 = private unnamed_addr constant [21 x i8] c"ext_constant_var_def\00"
// * constant unnamed string with GPU binary
// CUDA: @[[FATBIN:.*]] = private constant{{.*GPU binary would be here.*}}\00",
// HIPEF: @[[FATBIN:.*]] = private constant{{.*GPU binary would be here.*}}\00",{{.*}}align 4096
// HIPNEF: @[[FATBIN:__hip_fatbin]] = external constant i8, section ".hip_fatbin"
// CUDANORDC-SAME: section ".nv_fatbin", align 8
// CUDARDC-SAME: section "__nv_relfatbin", align 8
// * constant struct that wraps GPU binary
// ALL: @__[[PREFIX:cuda|hip]]_fatbin_wrapper = internal constant
// LNX-SAME: { i32, i32, i8*, i8* }
// CUDA-SAME: { i32 1180844977, i32 1,
// HIP-SAME: { i32 1212764230, i32 1,
// CUDA-SAME: i8* getelementptr inbounds ({{.*}}@[[FATBIN]], i64 0, i64 0),
// HIPEF-SAME: i8* getelementptr inbounds ({{.*}}@[[FATBIN]], i64 0, i64 0),
// HIPNEF-SAME:  i8* @[[FATBIN]],
// LNX-SAME: i8* null }
// CUDA-SAME: section ".nvFatBinSegment"
// HIP-SAME: section ".hipFatBinSegment"
// * variable to save GPU binary handle after initialization
// CUDANORDC: @__[[PREFIX]]_gpubin_handle = internal global i8** null
// HIPNEF: @__[[PREFIX]]_gpubin_handle = linkonce hidden global i8** null
// * constant unnamed string with NVModuleID
// CUDARDC: [[MODULE_ID_GLOBAL:@.*]] = private constant
// CUDARDC-SAME: c"[[MODULE_ID:.+]]\00", section "__nv_module_id", align 32
// * Make sure our constructor was added to global ctor list.
// LNX: @llvm.global_ctors = appending global {{.*}}@__[[PREFIX]]_module_ctor
// * Alias to global symbol containing the NVModuleID.
// CUDARDC: @__fatbinwrap[[MODULE_ID]] ={{.*}} alias { i32, i32, i8*, i8* }
// CUDARDC-SAME: { i32, i32, i8*, i8* }* @__[[PREFIX]]_fatbin_wrapper

// Test that we build the correct number of calls to cudaSetupArgument followed
// by a call to cudaLaunch.

// LNX: define{{.*}}kernelfunc

// New launch sequence stores arguments into local buffer and passes array of
// pointers to them directly to cudaLaunchKernel
// CUDA-NEW: alloca
// CUDA-NEW: store
// CUDA-NEW: store
// CUDA-NEW: store
// CUDA-NEW: call{{.*}}__cudaPopCallConfiguration
// CUDA-NEW: call{{.*}}cudaLaunchKernel

// Legacy style launch sequence sets up arguments by passing them to
// [cuda|hip]SetupArgument.
// CUDA-OLD: call{{.*}}[[PREFIX]]SetupArgument
// CUDA-OLD: call{{.*}}[[PREFIX]]SetupArgument
// CUDA-OLD: call{{.*}}[[PREFIX]]SetupArgument
// CUDA-OLD: call{{.*}}[[PREFIX]]Launch

// HIP: call{{.*}}[[PREFIX]]SetupArgument
// HIP: call{{.*}}[[PREFIX]]SetupArgument
// HIP: call{{.*}}[[PREFIX]]SetupArgument
// HIP: call{{.*}}[[PREFIX]]Launch
__global__ void kernelfunc(int i, int j, int k) {}

// Test that we've built correct kernel launch sequence.
// LNX: define{{.*}}hostfunc
// CUDA-OLD: call{{.*}}[[PREFIX]]ConfigureCall
// CUDA-NEW: call{{.*}}__cudaPushCallConfiguration
// HIP: call{{.*}}[[PREFIX]]ConfigureCall
// LNX: call{{.*}}kernelfunc
void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }
#endif

// Test that we've built a function to register kernels and global vars.
// ALL: define internal void @__[[PREFIX]]_register_globals
// ALL: call{{.*}}[[PREFIX]]RegisterFunction(i8** %0, {{.*}}kernelfunc{{[^,]*}}, {{[^@]*}}@0
// ALL-DAG: call void {{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}device_var{{[^,]*}}, {{[^@]*}}@1, {{.*}}i32 0, {{i32|i64}} 4, i32 0, i32 0
// ALL-DAG: call void {{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}constant_var{{[^,]*}}, {{[^@]*}}@2, {{.*}}i32 0, {{i32|i64}} 4, i32 1, i32 0
// ALL-DAG: call void {{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}ext_device_var_def{{[^,]*}}, {{[^@]*}}@3, {{.*}}i32 0, {{i32|i64}} 4, i32 0, i32 0
// ALL-DAG: call void {{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}ext_constant_var_def{{[^,]*}}, {{[^@]*}}@4, {{.*}}i32 0, {{i32|i64}} 4, i32 1, i32 0
// LNX_17-DAG: [[PREFIX]]RegisterVar(i8** %0, {{.*}}inline_var
// LNX_17-NOT: [[PREFIX]]RegisterVar(i8** %0, {{.*}}inline_var2
// ALL: ret void

// Test that we've built a constructor.
// LNX: define internal void @__[[PREFIX]]_module_ctor

// In separate mode it calls __[[PREFIX]]RegisterFatBinary(&__[[PREFIX]]_fatbin_wrapper)
// HIP only register fat binary once.
// HIP: load i8**, i8*** @__hip_gpubin_handle
// HIP-NEXT: icmp eq i8** {{.*}}, null
// HIP-NEXT: br i1 {{.*}}, label %if, label %exit
// HIP: if:
// CUDANORDC: call{{.*}}[[PREFIX]]RegisterFatBinary{{.*}}__[[PREFIX]]_fatbin_wrapper
//   .. stores return value in __[[PREFIX]]_gpubin_handle
// CUDANORDC-NEXT: store{{.*}}__[[PREFIX]]_gpubin_handle
//   .. and then calls __[[PREFIX]]_register_globals
// HIP: call{{.*}}[[PREFIX]]RegisterFatBinary{{.*}}__[[PREFIX]]_fatbin_wrapper
//   .. stores return value in __[[PREFIX]]_gpubin_handle
// HIP-NEXT: store{{.*}}__[[PREFIX]]_gpubin_handle
//   .. and then calls __[[PREFIX]]_register_globals
// HIP-NEXT: br label %exit
// HIP: exit:
// HIP-NEXT: load i8**, i8*** @__hip_gpubin_handle
// CUDANORDC-NEXT: call void @__[[PREFIX]]_register_globals
// HIP-NEXT: call void @__[[PREFIX]]_register_globals
// * In separate mode we also register a destructor.
// CUDANORDC-NEXT: call i32 @atexit(void (i8*)* @__[[PREFIX]]_module_dtor)
// HIP-NEXT: call i32 @atexit(void (i8*)* @__[[PREFIX]]_module_dtor)

// With relocatable device code we call __[[PREFIX]]RegisterLinkedBinary%NVModuleID%
// CUDARDC: call{{.*}}__[[PREFIX]]RegisterLinkedBinary[[MODULE_ID]](
// CUDARDC-SAME: __[[PREFIX]]_register_globals, {{.*}}__[[PREFIX]]_fatbin_wrapper
// CUDARDC-SAME: [[MODULE_ID_GLOBAL]]

// Test that we've created destructor.
// CUDANORDC: define internal void @__[[PREFIX]]_module_dtor
// HIP: define internal void @__[[PREFIX]]_module_dtor
// CUDANORDC: load{{.*}}__[[PREFIX]]_gpubin_handle
// HIP: load{{.*}}__[[PREFIX]]_gpubin_handle
// CUDANORDC-NEXT: call void @__[[PREFIX]]UnregisterFatBinary
// HIP-NEXT: icmp ne i8** {{.*}}, null
// HIP-NEXT: br i1 {{.*}}, label %if, label %exit
// HIP: if:
// HIP-NEXT: call void @__[[PREFIX]]UnregisterFatBinary
// HIP-NEXT: store i8** null, i8*** @__hip_gpubin_handle
// HIP-NEXT: br label %exit
// HIP: exit:

// There should be no __[[PREFIX]]_register_globals if we have no
// device-side globals, but we still need to register GPU binary.
// Skip GPU binary string first.
// CUDANOGLOBALS-NOT: @{{.*}} = private constant{{.*}}
// HIPNOGLOBALS-NOT: @{{.*}} = internal constant{{.*}}
// NOGLOBALS-NOT: define internal void @__{{.*}}_register_globals
// NOGLOBALS-NOT: define internal void @__{{cuda|hip}}_module_ctor
// NOGLOBALS-NOT: call{{.*}}{{cuda|hip}}RegisterFatBinary{{.*}}__{{cuda|hip}}_fatbin_wrapper
// NOGLOBALS-NOT: call void @__{{cuda|hip}}_register_globals
// NOGLOBALS-NOT: define internal void @__{{cuda|hip}}_module_dtor
// NOGLOBALS-NOT: call void @__{{cuda|hip}}UnregisterFatBinary

// There should be no constructors/destructors if we have no GPU binary.
// NOGPUBIN-NOT: define internal void @__{{cuda|hip}}_register_globals
// NOGPUBIN-NOT: define internal void @__{{cuda|hip}}_module_ctor
// NOGPUBIN-NOT: define internal void @__{{cuda|hip}}_module_dtor
