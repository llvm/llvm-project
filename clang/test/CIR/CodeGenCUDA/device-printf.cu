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

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir \
// RUN:            -emit-llvm %s -o %t-eh.ll
// RUN: FileCheck --check-prefix=EH --input-file=%t-eh.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_80 -x cuda \
// RUN:            -fcuda-is-device -fcxx-exceptions -fexceptions \
// RUN:            -emit-llvm %s -o %t-eh-og.ll
// RUN: FileCheck --check-prefix=EH --input-file=%t-eh-og.ll %s

__device__ void print_int() {
  printf("%d", 42);
}

// CIR: cir.func no_inline dso_local @_Z9print_intv()
// CIR:   %[[#VAL:]] = cir.const #cir.int<42> : !s32i
// CIR:   cir.offload.printf(%{{.+}}, %[[#VAL]]) : (!cir.ptr<!s8i>, !s32i) -> !s32i
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
// CIR:   cir.offload.printf(%{{.+}}) : (!cir.ptr<!s8i>) -> !s32i
// CIR:   cir.return

// LLVM: define dso_local void @_Z13print_no_argsv()
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr null)
// LLVM:   ret void

// The char is promoted to int. Each argument is stored with its own alignment.
__device__ void print_mixed(char c, double d, const char *s) {
  printf("%c %f %s", c, d, s);
}

// CIR: cir.func no_inline dso_local @_Z11print_mixedcdPKc(
// CIR:   cir.offload.printf(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!s8i>, !s32i, !cir.double, !cir.ptr<!s8i>) -> !s32i

// LLVM: define dso_local void @_Z11print_mixedcdPKc(
// LLVM:   %[[PACKED:.*]] = alloca {{%printf_args.*|. i32, double, ptr .}}, align 8
// LLVM:   %[[GEP0:.*]] = getelementptr inbounds nuw {{.*}}, ptr %[[PACKED]], i32 0, i32 0
// LLVM:   store i32 %{{.*}}, ptr %[[GEP0]], align 4
// LLVM:   %[[GEP1:.*]] = getelementptr inbounds nuw {{.*}}, ptr %[[PACKED]], i32 0, i32 1
// LLVM:   store double %{{.*}}, ptr %[[GEP1]], align 8
// LLVM:   %[[GEP2:.*]] = getelementptr inbounds nuw {{.*}}, ptr %[[PACKED]], i32 0, i32 2
// LLVM:   store ptr %{{.*}}, ptr %[[GEP2]], align 8
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr %[[PACKED]])

// A pointer to a data member is passed as its offset, which the C++ ABI
// lowering has substituted by the time CIR is emitted.
struct Pair {
  int first, second;
};

__device__ void print_member_pointer() {
  printf("%ld", &Pair::second);
}

// CIR: cir.func no_inline dso_local @_Z20print_member_pointerv()
// CIR:   %[[#OFFSET:]] = cir.const #cir.int<4> : !s64i
// CIR:   cir.offload.printf(%{{.+}}, %[[#OFFSET]]) : (!cir.ptr<!s8i>, !s64i) -> !s32i

// LLVM: define dso_local void @_Z20print_member_pointerv()
// LLVM:   %[[PACKED:.*]] = alloca {{%printf_args.*|. i64 .}}, align 8
// LLVM:   %[[GEP:.*]] = getelementptr inbounds nuw {{.*}}, ptr %[[PACKED]], i32 0, i32 0
// LLVM:   store i64 4, ptr %[[GEP]], align 8
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr %[[PACKED]])

typedef int int4v __attribute__((ext_vector_type(4)));

__device__ void print_vector(int4v v) {
  printf("%v4d", v);
}

// CIR: cir.func no_inline dso_local @_Z12print_vectorDv4_i(
// CIR:   cir.offload.printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i>, !cir.vector<4 x !s32i>) -> !s32i

// LLVM: define dso_local void @_Z12print_vectorDv4_i(
// LLVM:   %[[PACKED:.*]] = alloca {{%printf_args.*|. <4 x i32> .}}, align 16
// LLVM:   %[[GEP:.*]] = getelementptr inbounds nuw {{.*}}, ptr %[[PACKED]], i32 0, i32 0
// LLVM:   store <4 x i32> %{{.*}}, ptr %[[GEP]], align 16
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr %[[PACKED]])

// The buffer is allocated in the entry block, not on every iteration.
__device__ void print_in_loop(int n) {
  for (int i = 0; i < n; ++i)
    printf("%d", i);
}

// LLVM: define dso_local void @_Z13print_in_loopi(
// LLVM-NOT: br
// LLVM:   %[[PACKED:.*]] = alloca {{%printf_args.*|. i32 .}}, align
// LLVM:   br
// LLVM:   call i32 @vprintf(ptr @{{.*}}, ptr %[[PACKED]])

// vprintf is not called with invoke, even when a cleanup is active.
struct WithDtor {
  __device__ ~WithDtor();
};

__device__ void print_with_cleanup() {
  WithDtor w;
  printf("%d", 1);
}

// EH: define dso_local void @_Z18print_with_cleanupv() #{{[0-9]+}} {
// EH-NOT: invoke
// EH:   call i32 @vprintf(
// EH-NOT: landingpad
// EH:   call void @_ZN8WithDtorD1Ev(
// EH:   ret void
