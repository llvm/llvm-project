// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefix=LLVM

typedef unsigned long size_t;

void test_memcpy(void *dst, const void *src, size_t n) {
  __builtin_memcpy(dst, src, n);
}

// CIR-LABEL: cir.func no_inline dso_local @test_memcpy
// CIR:         %[[DST:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[SRC:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[N:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.libc.memcpy %[[N]] bytes from %[[SRC]] align(1) to %[[DST]] align(1) : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM-LABEL: define dso_local void @test_memcpy
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, i64 %{{.*}}, i1 false)

void *test_memcpy_ret(void *dst, const void *src, size_t n) {
  return __builtin_memcpy(dst, src, n);
}

// CIR-LABEL: cir.func no_inline dso_local @test_memcpy_ret
// CIR:         %[[DST:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[SRC:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[N:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.libc.memcpy %[[N]] bytes from %[[SRC]] align(1) to %[[DST]] align(1) : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
// CIR:         cir.return %{{.*}} : !cir.ptr<!void>

// LLVM-LABEL: define dso_local ptr @test_memcpy_ret
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, i64 %{{.*}}, i1 false)
// LLVM:         ret ptr %{{.*}}

void test_memcpy_int(int *dst, const int *src, size_t n) {
  __builtin_memcpy(dst, src, n);
}

// CIR-LABEL: cir.func no_inline dso_local @test_memcpy_int
// CIR:         %[[DST_I:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:    %[[DST:.*]] = cir.cast bitcast %[[DST_I]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[SRC_I:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:    %[[SRC:.*]] = cir.cast bitcast %[[SRC_I]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
// CIR-NEXT:    %[[N:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.libc.memcpy %[[N]] bytes from %[[SRC]] align(4) to %[[DST]] align(4) : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>

// LLVM-LABEL: define dso_local void @test_memcpy_int
// LLVM:         %[[DST:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-NEXT:    %[[SRC:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-NEXT:    %[[N:.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[DST]], ptr align 4 %[[SRC]], i64 %[[N]], i1 false)

void *test_mempcpy(void *dst, const void *src, size_t n) {
  return __builtin_mempcpy(dst, src, n);
}

// CIR-LABEL: cir.func no_inline dso_local @test_mempcpy
// CIR:         %[[DST:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[SRC:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT:    %[[N:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    cir.libc.memcpy %[[N]] bytes from %[[SRC]] align(1) to %[[DST]] align(1) : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
// CIR-NEXT:    %[[END:.*]] = cir.ptr_stride %[[DST]], %[[N]] : (!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
// CIR:         cir.return %{{.*}} : !cir.ptr<!void>

// LLVM-LABEL: define dso_local ptr @test_mempcpy
// LLVM:         %[[DST:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-NEXT:    %[[SRC:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-NEXT:    %[[N:.*]] = load i64, ptr %{{.*}}, align 8
// LLVM-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[DST]], ptr align 1 %[[SRC]], i64 %[[N]], i1 false)
// LLVM-NEXT:    getelementptr{{.*}} i8, ptr %[[DST]], i64 %[[N]]
// Note: OG emits getelementptr inbounds; CIR omits inbounds (missing feature in cir.ptr_stride lowering).
