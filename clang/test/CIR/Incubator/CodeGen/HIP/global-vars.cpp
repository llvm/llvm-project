#include "cuda.h"

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ -x hip %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ -x hip %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x hip -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/../Inputs/ -x hip %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

__shared__ int shared;
// CIR-DEVICE: cir.global external{{.*}}lang_address_space(offload_local) @shared = #cir.undef
// LLVM-DEVICE: @shared = addrspace(3) global i32 undef, align 4
// CIR-HOST: cir.global{{.*}}@shared = #cir.undef : !s32i {alignment = 4 : i64}
// CIR-HOST-NOT: cu.shadow_name
// LLVM-HOST: @shared = internal global i32 undef, align 4
// OGCG-HOST: @shared = internal global i32
// OGCG-DEVICE: @shared = addrspace(3) global i32 undef, align 4

__constant__ int b;
// CIR-DEVICE: cir.global constant external{{.*}}lang_address_space(offload_constant) @b = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized, cu.var_registration = #cir.cu.var_registration<Variable, constant>}
// LLVM-DEVICE: @b = addrspace(4) externally_initialized constant i32 0, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@b = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<b>, cu.var_registration = #cir.cu.var_registration<Variable, constant>}
// LLVM-HOST: @b = internal global i32 undef, align 4
// OGCG-HOST: @b = internal global i32
// OGCG-DEVICE: @b = addrspace(4) externally_initialized constant i32 0, align 4

// External device variables should remain external on host side (they're just declarations)
// Note: External declarations may not appear in output if they're not referenced
extern __device__ int ext_device_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_device_var
// LLVM-HOST-NOT: @ext_device_var
// OGCG-HOST-NOT: @ext_device_var
// OGCG-DEVICE-NOT: @ext_device_var

extern __constant__ int ext_constant_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_constant_var
// LLVM-HOST-NOT: @ext_constant_var
// OGCG-HOST-NOT: @ext_constant_var
// OGCG-DEVICE-NOT: @ext_constant_var

// External device variables with definitions should be internal on host
extern __device__ int ext_device_var_def;
__device__ int ext_device_var_def = 1;
// CIR-DEVICE: cir.global external{{.*}}lang_address_space(offload_global) @ext_device_var_def = #cir.int<1>
// LLVM-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_device_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_device_var_def>, cu.var_registration = #cir.cu.var_registration<Variable>}
// LLVM-HOST: @ext_device_var_def = internal global i32 undef, align 4
// OGCG-HOST: @ext_device_var_def = internal global i32
// OGCG-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4

extern __constant__ int ext_constant_var_def;
__constant__ int ext_constant_var_def = 2;
// CIR-DEVICE: cir.global constant external{{.*}}lang_address_space(offload_constant) @ext_constant_var_def = #cir.int<2>
// LLVM-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// OGCG-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_constant_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_constant_var_def>, cu.var_registration = #cir.cu.var_registration<Variable, constant>}
// LLVM-HOST: @ext_constant_var_def = internal global i32 undef, align 4
// OGCG-HOST: @ext_constant_var_def = internal global i32

// Regular host variables should NOT be internalized
int host_var;
// CIR-HOST: cir.global external @host_var = #cir.int<0> : !s32i
// LLVM-HOST: @host_var = global i32 0, align 4
// OGCG-HOST: @host_var ={{.*}} global i32

// CIR-DEVICE-NOT: cir.global{{.*}}@host_var
// LLVM-DEVICE-NOT: @host_var
// OGCG-DEVICE-NOT: @host_var

// External host variables should remain external (may not appear if not referenced)
extern int ext_host_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_host_var
// LLVM-HOST-NOT: @ext_host_var
// OGCG-HOST-NOT: @ext_host_var

// CIR-DEVICE-NOT: cir.global{{.*}}@ext_host_var
// LLVM-DEVICE-NOT: @ext_host_var
// OGCG-DEVICE-NOT: @ext_host_var
