#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-og.ll %s

__device__ void print_int() {
  printf("%d", 42);
}

// CIR: cir.func no_inline dso_local @_Z9print_intv()
// CIR:   %[[#ALLOCA:]] = cir.alloca !rec_anon_struct
// CIR:   %[[#VAL:]] = cir.const #cir.int<42> : !s32i
// CIR:   %[[#FIELD:]] = cir.get_member %[[#ALLOCA]][0]
// CIR:   cir.store align(4) %[[#VAL]], %[[#FIELD]]
// CIR:   %[[#BUF:]] = cir.cast bitcast %[[#ALLOCA]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!void>
// CIR:   cir.call @vprintf(%{{.+}}, %[[#BUF]])
// CIR:   cir.return

// LLVM: define dso_local void @_Z9print_intv()
// LLVM:   %[[PACKED:.*]] = alloca
// LLVM:   %[[GEP:.*]] = getelementptr inbounds nuw
// LLVM:   store i32 42, ptr %[[GEP]], align 4
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr %[[PACKED]])
// LLVM:   ret void

__device__ void print_no_args() {
  printf("hello world");
}

// CIR: cir.func no_inline dso_local @_Z13print_no_argsv()
// CIR:   %[[#NULL:]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:   cir.call @vprintf(%{{.+}}, %[[#NULL]])
// CIR:   cir.return

// LLVM: define dso_local void @_Z13print_no_argsv()
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr null)
// LLVM:   ret void
