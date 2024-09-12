// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

global int a = 13;
// CIR-DAG: cir.global external addrspace(offload_global) @a = #cir.int<13> : !s32i
// LLVM-DAG: @a = addrspace(1) global i32 13

global int b = 15;
// CIR-DAG: cir.global external addrspace(offload_global) @b = #cir.int<15> : !s32i
// LLVM-DAG: @b = addrspace(1) global i32 15

kernel void test_get_global() {
  a = b;
  // CIR:      %[[#ADDRB:]] = cir.get_global @b : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-NEXT: %[[#LOADB:]] = cir.load %[[#ADDRB]] : !cir.ptr<!s32i, addrspace(offload_global)>, !s32i
  // CIR-NEXT: %[[#ADDRA:]] = cir.get_global @a : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-NEXT: cir.store %[[#LOADB]], %[[#ADDRA]] : !s32i, !cir.ptr<!s32i, addrspace(offload_global)>

  // LLVM:      %[[#LOADB:]] = load i32, ptr addrspace(1) @b, align 4
  // LLVM-NEXT: store i32 %[[#LOADB]], ptr addrspace(1) @a, align 4
}
