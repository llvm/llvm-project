// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

kernel void test_static(int i) {
  static global int b = 15;
  // CIR-DAG: cir.global "private" internal dsolocal addrspace(offload_global) @test_static.b = #cir.int<15> : !s32i {alignment = 4 : i64}
  // LLVM-DAG: @test_static.b = internal addrspace(1) global i32 15

  local int c;
  // CIR-DAG: cir.global "private" internal dsolocal addrspace(offload_local) @test_static.c : !s32i {alignment = 4 : i64}
  // LLVM-DAG: @test_static.c = internal addrspace(3) global i32 undef

  // CIR-DAG: %[[#ADDRB:]] = cir.get_global @test_static.b : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-DAG: %[[#ADDRC:]] = cir.get_global @test_static.c : !cir.ptr<!s32i, addrspace(offload_local)>

  c = b;
  // CIR:      %[[#LOADB:]] = cir.load %[[#ADDRB]] : !cir.ptr<!s32i, addrspace(offload_global)>, !s32i
  // CIR-NEXT: cir.store %[[#LOADB]], %[[#ADDRC]] : !s32i, !cir.ptr<!s32i, addrspace(offload_local)>

  // LLVM:     %[[#LOADB:]] = load i32, ptr addrspace(1) @test_static.b, align 4
  // LLVM-NEXT: store i32 %[[#LOADB]], ptr addrspace(3) @test_static.c, align 4
}
