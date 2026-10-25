// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -Wno-frame-address -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -Wno-frame-address -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -Wno-frame-address -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// The frame address is a private (AS 5) pointer that must be cast to the
// generic void * result type.

void *test_frame_address() {
  return __builtin_frame_address(1);
}

// CIR-LABEL: cir.func{{.*}} @_Z18test_frame_addressv()
// CIR:         %[[ARG:.*]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT:    %[[ADDR:.*]] = cir.frame_address(%[[ARG]]) : !cir.ptr<!u8i, target_address_space(5)>
// CIR-NEXT:    %[[VOID:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i, target_address_space(5)> -> !cir.ptr<!void, target_address_space(5)>
// CIR-NEXT:    %[[CAST:.*]] = cir.cast address_space %[[VOID]] : !cir.ptr<!void, target_address_space(5)> -> !cir.ptr<!void>
// CIR-NEXT:    cir.store %[[CAST]], %{{.*}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>, target_address_space(5)>

// LLVM-LABEL: define{{.*}} ptr @_Z18test_frame_addressv()
// LLVM:         %[[ADDR:.*]] = call ptr addrspace(5) @llvm.frameaddress.p5(i32 1)
// LLVM-NEXT:    %[[CAST:.*]] = addrspacecast ptr addrspace(5) %[[ADDR]] to ptr
// LLVM-NEXT:    store ptr %[[CAST]], ptr addrspace(5) %{{.*}}, align 8

// OGCG-LABEL: define{{.*}} ptr @_Z18test_frame_addressv()
// OGCG:         %[[ADDR:.*]] = call ptr addrspace(5) @llvm.frameaddress.p5(i32 1)
// OGCG-NEXT:    store ptr addrspace(5) %[[ADDR]], ptr addrspace(5) %{{.*}}, align 8
