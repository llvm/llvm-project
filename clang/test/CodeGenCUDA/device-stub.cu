// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s --check-prefixes=ALL,NORDC,CUDA,CUDANORDC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o -  -DNOGLOBALS \
// RUN:   | FileCheck %s -check-prefixes=NOGLOBALS,CUDANOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-rdc -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s --check-prefixes=ALL,RDC,CUDA,CUDARDC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=NOGPUBIN

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck %s --check-prefixes=ALL,NORDC,HIP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o -  -DNOGLOBALS -x hip \
// RUN:   | FileCheck %s -check-prefixes=NOGLOBALS,HIPNOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-rdc -fcuda-include-gpubinary %t -o - -x hip \
// RUN:   | FileCheck %s --check-prefixes=ALL,RDC,HIP,HIPRDC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip\
// RUN:   | FileCheck %s -check-prefix=NOGPUBIN

#include "Inputs/cuda.h"

#ifndef NOGLOBALS
// ALL-DAG: @device_var = internal global i32
__device__ int device_var;

// ALL-DAG: @constant_var = internal global i32
__constant__ int constant_var;

// ALL-DAG: @shared_var = internal global i32
__shared__ int shared_var;

// Make sure host globals don't get internalized...
// ALL-DAG: @host_var = global i32
int host_var;
// ... and that extern vars remain external.
// ALL-DAG: @ext_host_var = external global i32
extern int ext_host_var;

// Shadows for external device-side variables are *definitions* of
// those variables.
// ALL-DAG: @ext_device_var = internal global i32
extern __device__ int ext_device_var;
// ALL-DAG: @ext_device_var = internal global i32
extern __constant__ int ext_constant_var;

void use_pointers() {
  int *p;
  p = &device_var;
  p = &constant_var;
  p = &shared_var;
  p = &host_var;
  p = &ext_device_var;
  p = &ext_constant_var;
  p = &ext_host_var;
}

// Make sure that all parts of GPU code init/cleanup are there:
// * constant unnamed string with the kernel name
// ALL: private unnamed_addr constant{{.*}}kernelfunc{{.*}}\00"
// * constant unnamed string with GPU binary
// HIP: @[[FATBIN:__hip_fatbin]] = external constant i8, section ".hip_fatbin"
// CUDA: @[[FATBIN:.*]] = private constant{{.*GPU binary would be here.*}}\00",
// CUDANORDC-SAME: section ".nv_fatbin", align 8
// CUDARDC-SAME: section "__nv_relfatbin", align 8
// * constant struct that wraps GPU binary
// ALL: @__[[PREFIX:cuda|hip]]_fatbin_wrapper = internal constant
// ALL-SAME: { i32, i32, i8*, i8* }
// CUDA-SAME: { i32 1180844977, i32 1,
// HIP-SAME: { i32 1212764230, i32 1,
// CUDA-SAME: i8* getelementptr inbounds ({{.*}}@[[FATBIN]], i64 0, i64 0),
// HIP-SAME:  i8* @[[FATBIN]],
// ALL-SAME: i8* null }
// CUDA-SAME: section ".nvFatBinSegment"
// HIP-SAME: section ".hipFatBinSegment"
// * variable to save GPU binary handle after initialization
// NORDC: @__[[PREFIX]]_gpubin_handle = internal global i8** null
// * constant unnamed string with NVModuleID
// RDC: [[MODULE_ID_GLOBAL:@.*]] = private constant
// CUDARDC-SAME: c"[[MODULE_ID:.+]]\00", section "__nv_module_id", align 32
// HIPRDC-SAME: c"[[MODULE_ID:.+]]\00", section "__hip_module_id", align 32
// * Make sure our constructor was added to global ctor list.
// ALL: @llvm.global_ctors = appending global {{.*}}@__[[PREFIX]]_module_ctor
// * In separate mode we also register a destructor.
// NORDC: @llvm.global_dtors = appending global {{.*}}@__[[PREFIX]]_module_dtor
// * Alias to global symbol containing the NVModuleID.
// RDC: @__fatbinwrap[[MODULE_ID]] = alias { i32, i32, i8*, i8* }
// RDC-SAME: { i32, i32, i8*, i8* }* @__[[PREFIX]]_fatbin_wrapper

// Test that we build the correct number of calls to cudaSetupArgument followed
// by a call to cudaLaunch.

// ALL: define{{.*}}kernelfunc
// ALL: call{{.*}}[[PREFIX]]SetupArgument
// ALL: call{{.*}}[[PREFIX]]SetupArgument
// ALL: call{{.*}}[[PREFIX]]SetupArgument
// ALL: call{{.*}}[[PREFIX]]Launch
__global__ void kernelfunc(int i, int j, int k) {}

// Test that we've built correct kernel launch sequence.
// ALL: define{{.*}}hostfunc
// ALL: call{{.*}}[[PREFIX]]ConfigureCall
// ALL: call{{.*}}kernelfunc
void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }
#endif

// Test that we've built a function to register kernels and global vars.
// ALL: define internal void @__[[PREFIX]]_register_globals
// ALL: call{{.*}}[[PREFIX]]RegisterFunction(i8** %0, {{.*}}kernelfunc
// ALL-DAG: call{{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}device_var{{.*}}i32 0, i32 4, i32 0, i32 0
// ALL-DAG: call{{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}constant_var{{.*}}i32 0, i32 4, i32 1, i32 0
// ALL-DAG: call{{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}ext_device_var{{.*}}i32 1, i32 4, i32 0, i32 0
// ALL-DAG: call{{.*}}[[PREFIX]]RegisterVar(i8** %0, {{.*}}ext_constant_var{{.*}}i32 1, i32 4, i32 1, i32 0
// ALL: ret void

// Test that we've built a constructor.
// ALL: define internal void @__[[PREFIX]]_module_ctor

// In separate mode it calls __[[PREFIX]]RegisterFatBinary(&__[[PREFIX]]_fatbin_wrapper)
// NORDC: call{{.*}}[[PREFIX]]RegisterFatBinary{{.*}}__[[PREFIX]]_fatbin_wrapper
//   .. stores return value in __[[PREFIX]]_gpubin_handle
// NORDC-NEXT: store{{.*}}__[[PREFIX]]_gpubin_handle
//   .. and then calls __[[PREFIX]]_register_globals
// NORDC-NEXT: call void @__[[PREFIX]]_register_globals

// With relocatable device code we call __[[PREFIX]]RegisterLinkedBinary%NVModuleID%
// RDC: call{{.*}}__[[PREFIX]]RegisterLinkedBinary[[MODULE_ID]](
// RDC-SAME: __[[PREFIX]]_register_globals, {{.*}}__[[PREFIX]]_fatbin_wrapper
// RDC-SAME: [[MODULE_ID_GLOBAL]]

// Test that we've created destructor.
// NORDC: define internal void @__[[PREFIX]]_module_dtor
// NORDC: load{{.*}}__[[PREFIX]]_gpubin_handle
// NORDC-NEXT: call void @__[[PREFIX]]UnregisterFatBinary

// There should be no __[[PREFIX]]_register_globals if we have no
// device-side globals, but we still need to register GPU binary.
// Skip GPU binary string first.
// CUDANOGLOBALS: @{{.*}} = private constant{{.*}}
// HIPNOGLOBALS: @{{.*}} = external constant{{.*}}
// NOGLOBALS-NOT: define internal void @__{{.*}}_register_globals
// NOGLOBALS: define internal void @__[[PREFIX:cuda|hip]]_module_ctor
// NOGLOBALS: call{{.*}}[[PREFIX]]RegisterFatBinary{{.*}}__[[PREFIX]]_fatbin_wrapper
// NOGLOBALS-NOT: call void @__[[PREFIX]]_register_globals
// NOGLOBALS: define internal void @__[[PREFIX]]_module_dtor
// NOGLOBALS: call void @__[[PREFIX]]UnregisterFatBinary

// There should be no constructors/destructors if we have no GPU binary.
// NOGPUBIN-NOT: define internal void @__[[PREFIX]]_register_globals
// NOGPUBIN-NOT: define internal void @__[[PREFIX]]_module_ctor
// NOGPUBIN-NOT: define internal void @__[[PREFIX]]_module_dtor
