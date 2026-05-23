#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir \
// RUN:   -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t-pre.cir
// RUN: FileCheck --check-prefix=CIR-PRE --input-file=%t-pre.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-POST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -I%S/Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.cir %s

// Verifies CIR emits correct address spaces for CUDA globals.

// CIR-DEVICE: cir.global "private" internal dso_local @_ZZ2fnvE1j = #cir.undef
// LLVM-DEVICE: @_ZZ2fnvE1j = internal global i32 undef

// CIR-PRE: cir.global external  lang_address_space(offload_global) @i = #cir.int<0>
// CIR-POST: cir.global external  target_address_space(1) @i = #cir.int<0>
// LLVM-DEVICE-DAG: @i = addrspace(1) {{.*}}global i32 0
// OGCG-DAG: @i = addrspace(1) externally_initialized global i32 0
// CIR-HOST: cir.global {{.*}} @i = #cir.poison : {{.*}} {{{.*}}, cu.var_registration = #cir.cu.var_registration<i, Variable>}
// LLVM-HOST: @i = internal global i32 poison
// OGCG-HOST: @i = internal global i32 undef
__device__ int i;

// CIR-PRE: cir.global constant external  lang_address_space(offload_constant) @j = #cir.int<0>
// CIR-POST: cir.global constant external  target_address_space(4) @j = #cir.int<0>
// CIR-DEVICE: cir.global constant external target_address_space(4) @j = #cir.int<0> : {{.*}} {{{.*}}, cu.externally_initialized = #cir.cu.externally_initialized, cu.var_registration = #cir.cu.var_registration<j, Variable, constant>}
// LLVM-DEVICE-DAG: @j = addrspace(4) {{.*}}constant i32 0
// OGCG-DAG: @j = addrspace(4) externally_initialized constant i32 0
// CIR-HOST:  cir.global {{.*}} @j = #cir.poison : {{.*}} {{{.*}}, cu.var_registration = #cir.cu.var_registration<j, Variable, constant>}
// LLVM-HOST: @j = internal global i32 poison
// OGCG-HOST: @j = internal global i32 undef
__constant__ int j;

// CIR-PRE: cir.global external  lang_address_space(offload_local) @k = #cir.poison
// CIR-POST: cir.global external  target_address_space(3) @k = #cir.poison
// CIR-DEVICE: cir.global external target_address_space(3) @k = #cir.poison
// LLVM-DEVICE-DAG: @k = addrspace(3) global i32 {{undef|poison}}
// OGCG-DAG: @k = addrspace(3) global i32 undef
// CIR-HOST: cir.global {{.*}} @k = #cir.poison
// LLVM-HOST: @k = internal global i32 poison
// OGCG-HOST: @k = internal global i32 undef
__shared__ int k;

// CIR-PRE: cir.global external  lang_address_space(offload_local) @b = #cir.poison : !cir.float
// CIR-POST: cir.global external  target_address_space(3) @b = #cir.poison : !cir.float
// LLVM-DEVICE-DAG: @b = addrspace(3) global float {{undef|poison}}
// OGCG-DAG: @b = addrspace(3) global float undef
__shared__ float b;

// External device variables should remain external on host side (they're just declarations)
// Note: External declarations may not appear in output if they're not referenced
// CIR-HOST-NOT: cir.global{{.*}}@ext_device_var
// LLVM-HOST-NOT: @ext_device_var
// OGCG-HOST-NOT: @ext_device_var
// OGCG-DEVICE-NOT: @ext_device_var
extern __device__ int ext_device_var;

// CIR-HOST-NOT: cir.global{{.*}}@ext_constant_var
// LLVM-HOST-NOT: @ext_constant_var
// OGCG-HOST-NOT: @ext_constant_var
// OGCG-DEVICE-NOT: @ext_constant_var
extern __constant__ int ext_constant_var;

// External device variables with definitions should be internal on host
// CIR-DEVICE: cir.global external target_address_space(1) @ext_device_var_def = #cir.int<1>
// LLVM-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
// CIR-HOST: cir.global "private" internal {{.*}} @ext_device_var_def = #cir.poison : {{.*}} {{{.*}}, cu.var_registration = #cir.cu.var_registration<ext_device_var_def, Variable>}
// LLVM-HOST: @ext_device_var_def = internal global i32 poison, align 4
// OGCG-HOST: @ext_device_var_def = internal global i32
// OGCG-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
extern __device__ int ext_device_var_def;
__device__ int ext_device_var_def = 1;

// CIR-DEVICE: cir.global constant external target_address_space(4) @ext_constant_var_def = #cir.int<2>
// LLVM-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// OGCG-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// CIR-HOST: cir.global "private" internal {{.*}} @ext_constant_var_def = #cir.poison : {{.*}} {{{.*}}, cu.var_registration = #cir.cu.var_registration<ext_constant_var_def, Variable, constant>}
// LLVM-HOST: @ext_constant_var_def = internal global i32 poison, align 4
// OGCG-HOST: @ext_constant_var_def = internal global i32
extern __constant__ int ext_constant_var_def;
__constant__ int ext_constant_var_def = 2;

// Regular host variables should NOT be internalized
// CIR-HOST: cir.global external @host_var = #cir.int<0> : !s32i
// LLVM-HOST: @host_var = global i32 0, align 4
// OGCG-HOST: @host_var ={{.*}} global i32

// CIR-DEVICE-NOT: cir.global{{.*}}@host_var
// LLVM-DEVICE-NOT: @host_var
// OGCG-DEVICE-NOT: @host_var
int host_var;

// External host variables should remain external (may not appear if not referenced)
// CIR-HOST-NOT: cir.global{{.*}}@ext_host_var
// LLVM-HOST-NOT: @ext_host_var
// OGCG-HOST-NOT: @ext_host_var

// CIR-DEVICE-NOT: cir.global{{.*}}@ext_host_var
// LLVM-DEVICE-NOT: @ext_host_var
// OGCG-DEVICE-NOT: @ext_host_var
extern int ext_host_var;


__device__ void foo() {
  // CIR-PRE: cir.get_global @i : !cir.ptr<!s32i, lang_address_space(offload_global)>
  // CIR-POST: cir.get_global @i : !cir.ptr<!s32i, target_address_space(1)>
  i++;

  // CIR-PRE: cir.get_global @j : !cir.ptr<!s32i, lang_address_space(offload_constant)>
  // CIR-POST: cir.get_global @j : !cir.ptr<!s32i, target_address_space(4)>
  j++;

  // CIR-PRE: cir.get_global @k : !cir.ptr<!s32i, lang_address_space(offload_local)>
  // CIR-POST: cir.get_global @k : !cir.ptr<!s32i, target_address_space(3)>
  k++;
}

__global__ void fn() {
  int i = 0;
  __shared__ int j;
  j = i;
}

// CIR-DEVICE: cir.func {{.*}}@_Z2fnv() {{.*}} {
// CIR-DEVICE:   %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR-DEVICE:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   %[[J:.*]] = cir.get_global @_ZZ2fnvE1j : !cir.ptr<!s32i>
// CIR-DEVICE:   %[[VAL:.*]] = cir.load {{.*}}%[[ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-DEVICE:   cir.store {{.*}}%[[VAL]], %[[J]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:   cir.return

// LLVM-DEVICE: define dso_local ptx_kernel void @_Z2fnv()
// LLVM-DEVICE:   %[[ALLOCA:.*]] = alloca i32, i64 1, align 4
// LLVM-DEVICE:   store i32 0, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[ALLOCA]], align 4
// LLVM-DEVICE:   store i32 %[[VAL]], ptr @_ZZ2fnvE1j, align 4
// LLVM-DEVICE:   ret void

// OGCG-DEVICE: define dso_local ptx_kernel void @_Z2fnv()
// OGCG-DEVICE: entry:
// OGCG-DEVICE:   %[[ALLOCA:.*]] = alloca i32, align 4
// OGCG-DEVICE:   store i32 0, ptr %[[ALLOCA]], align 4
// OGCG-DEVICE:   %[[VAL:.*]] = load i32, ptr %[[ALLOCA]], align 4
// OGCG-DEVICE:   store i32 %[[VAL]], ptr addrspacecast (ptr addrspace(3) @_ZZ2fnvE1j to ptr), align 4
// OGCG-DEVICE:   ret void
