// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s --check-prefixes=ALL,NORDC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o -  -DNOGLOBALS \
// RUN:   | FileCheck %s -check-prefix=NOGLOBALS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-rdc -fcuda-include-gpubinary %t -o - \
// RUN:   | FileCheck %s --check-prefixes=ALL,RDC
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - \
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
// ALL: private unnamed_addr constant{{.*GPU binary would be here.*}}\00"
// NORDC-SAME: section ".nv_fatbin", align 8
// RDC-SAME: section "__nv_relfatbin", align 8
// * constant struct that wraps GPU binary
// ALL: @__cuda_fatbin_wrapper = internal constant { i32, i32, i8*, i8* } 
// ALL-SAME: { i32 1180844977, i32 1, {{.*}}, i8* null }
// ALL-SAME: section ".nvFatBinSegment"
// * variable to save GPU binary handle after initialization
// NORDC: @__cuda_gpubin_handle = internal global i8** null
// * constant unnamed string with NVModuleID
// RDC: [[MODULE_ID_GLOBAL:@.*]] = private unnamed_addr constant
// RDC-SAME: c"[[MODULE_ID:.+]]\00", section "__nv_module_id", align 32
// * Make sure our constructor was added to global ctor list.
// ALL: @llvm.global_ctors = appending global {{.*}}@__cuda_module_ctor
// * In separate mode we also register a destructor.
// NORDC: @llvm.global_dtors = appending global {{.*}}@__cuda_module_dtor
// * Alias to global symbol containing the NVModuleID.
// RDC: @__fatbinwrap[[MODULE_ID]] = alias { i32, i32, i8*, i8* }
// RDC-SAME: { i32, i32, i8*, i8* }* @__cuda_fatbin_wrapper

// Test that we build the correct number of calls to cudaSetupArgument followed
// by a call to cudaLaunch.

// ALL: define{{.*}}kernelfunc
// ALL: call{{.*}}cudaSetupArgument
// ALL: call{{.*}}cudaSetupArgument
// ALL: call{{.*}}cudaSetupArgument
// ALL: call{{.*}}cudaLaunch
__global__ void kernelfunc(int i, int j, int k) {}

// Test that we've built correct kernel launch sequence.
// ALL: define{{.*}}hostfunc
// ALL: call{{.*}}cudaConfigureCall
// ALL: call{{.*}}kernelfunc
void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }
#endif

// Test that we've built a function to register kernels and global vars.
// ALL: define internal void @__cuda_register_globals
// ALL: call{{.*}}cudaRegisterFunction(i8** %0, {{.*}}kernelfunc
// ALL-DAG: call{{.*}}cudaRegisterVar(i8** %0, {{.*}}device_var{{.*}}i32 0, i32 4, i32 0, i32 0
// ALL-DAG: call{{.*}}cudaRegisterVar(i8** %0, {{.*}}constant_var{{.*}}i32 0, i32 4, i32 1, i32 0
// ALL-DAG: call{{.*}}cudaRegisterVar(i8** %0, {{.*}}ext_device_var{{.*}}i32 1, i32 4, i32 0, i32 0
// ALL-DAG: call{{.*}}cudaRegisterVar(i8** %0, {{.*}}ext_constant_var{{.*}}i32 1, i32 4, i32 1, i32 0
// ALL: ret void

// Test that we've built a constructor.
// ALL: define internal void @__cuda_module_ctor

// In separate mode it calls __cudaRegisterFatBinary(&__cuda_fatbin_wrapper)
// NORDC: call{{.*}}cudaRegisterFatBinary{{.*}}__cuda_fatbin_wrapper
//   .. stores return value in __cuda_gpubin_handle
// NORDC-NEXT: store{{.*}}__cuda_gpubin_handle
//   .. and then calls __cuda_register_globals
// NORDC-NEXT: call void @__cuda_register_globals

// With relocatable device code we call __cudaRegisterLinkedBinary%NVModuleID%
// RDC: call{{.*}}__cudaRegisterLinkedBinary[[MODULE_ID]](
// RDC-SAME: __cuda_register_globals, {{.*}}__cuda_fatbin_wrapper
// RDC-SAME: [[MODULE_ID_GLOBAL]]

// Test that we've created destructor.
// NORDC: define internal void @__cuda_module_dtor
// NORDC: load{{.*}}__cuda_gpubin_handle
// NORDC-NEXT: call void @__cudaUnregisterFatBinary

// There should be no __cuda_register_globals if we have no
// device-side globals, but we still need to register GPU binary.
// Skip GPU binary string first.
// NOGLOBALS: @0 = private unnamed_addr constant{{.*}}
// NOGLOBALS-NOT: define internal void @__cuda_register_globals
// NOGLOBALS: define internal void @__cuda_module_ctor
// NOGLOBALS: call{{.*}}cudaRegisterFatBinary{{.*}}__cuda_fatbin_wrapper
// NOGLOBALS-NOT: call void @__cuda_register_globals
// NOGLOBALS: define internal void @__cuda_module_dtor
// NOGLOBALS: call void @__cudaUnregisterFatBinary

// There should be no constructors/destructors if we have no GPU binary.
// NOGPUBIN-NOT: define internal void @__cuda_register_globals
// NOGPUBIN-NOT: define internal void @__cuda_module_ctor
// NOGPUBIN-NOT: define internal void @__cuda_module_dtor
