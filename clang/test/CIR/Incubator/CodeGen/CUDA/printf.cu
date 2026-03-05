#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

__device__ void printer() {
  printf("%d", 0);
}

// CIR-DEVICE: cir.func {{.*}} @_Z7printerv() extra({{.*}}) {
// CIR-DEVICE:   %[[#Packed:]] = cir.alloca !rec_anon_struct
// CIR-DEVICE:   %[[#Zero:]] = cir.const #cir.int<0> : !s32i loc(#loc5)
// CIR-DEVICE:   %[[#Field0:]] = cir.get_member %0[0]
// CIR-DEVICE:   cir.store align(4) %[[#Zero]], %[[#Field0]]
// CIR-DEVICE:   %[[#Output:]] = cir.cast bitcast %[[#Packed]] : !cir.ptr<!rec_anon_struct>
// CIR-DEVICE:   cir.call @vprintf(%{{.+}}, %[[#Output]])
// CIR-DEVICE:   cir.return
// CIR-DEVICE: }

// LLVM-DEVICE: define dso_local void @_Z7printerv() {{.*}} {
// LLVM-DEVICE:   %[[#LLVMPacked:]] = alloca { i32 }, i64 1, align 8
// LLVM-DEVICE:   %[[#LLVMField0:]] = getelementptr { i32 }, ptr %[[#LLVMPacked]], i32 0, i32 0
// LLVM-DEVICE:   store i32 0, ptr %[[#LLVMField0]], align 4
// LLVM-DEVICE:   call i32 @vprintf(ptr @.str, ptr %[[#LLVMPacked]])
// LLVM-DEVICE:   ret void
// LLVM-DEVICE: }

__device__ void no_extra() {
  printf("hello world");
}

// CIR-DEVICE: cir.func {{.*}} @_Z8no_extrav() extra(#fn_attr) {
// CIR-DEVICE:   %[[#NULLPTR:]] = cir.const #cir.ptr<null>
// CIR-DEVICE:   cir.call @vprintf(%{{.+}}, %[[#NULLPTR]])
// CIR-DEVICE:   cir.return
// CIR-DEVICE: }

// LLVM-DEVICE: define dso_local void @_Z8no_extrav() {{.*}} {
// LLVM-DEVICE:   call i32 @vprintf(ptr @.str.1, ptr null)
// LLVM-DEVICE:   ret void
// LLVM-DEVICE: }
