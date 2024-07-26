// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM


// CIR: cir.func @func(%arg0: !cir.ptr<!s32i, addrspace(offload_local)>
// LLVM: @func(ptr addrspace(3)
kernel void func(local int *p) {
  // CIR-NEXT: %[[#ALLOCA_P:]] = cir.alloca !cir.ptr<!s32i, addrspace(offload_local)>, !cir.ptr<!cir.ptr<!s32i, addrspace(offload_local)>>, ["p", init] {alignment = 8 : i64}
  // LLVM-NEXT: %[[#ALLOCA_P:]] = alloca ptr addrspace(3), i64 1, align 8

  int x;
  // CIR-NEXT: %[[#ALLOCA_X:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
  // LLVM-NEXT: %[[#ALLOCA_X:]] = alloca i32, i64 1, align 4

  global char *b;
  // CIR-NEXT: %[[#ALLOCA_B:]] = cir.alloca !cir.ptr<!s8i, addrspace(offload_global)>, !cir.ptr<!cir.ptr<!s8i, addrspace(offload_global)>>, ["b"] {alignment = 8 : i64}
  // LLVM-NEXT: %[[#ALLOCA_B:]] = alloca ptr addrspace(1), i64 1, align 8

  // Store of the argument `p`
  // CIR-NEXT: cir.store %arg0, %[[#ALLOCA_P]] : !cir.ptr<!s32i, addrspace(offload_local)>, !cir.ptr<!cir.ptr<!s32i, addrspace(offload_local)>>
  // LLVM-NEXT: store ptr addrspace(3) %{{[0-9]+}}, ptr %[[#ALLOCA_P]], align 8

  return;
}
