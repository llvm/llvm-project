// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

void foo(void *a) {
  __builtin_prefetch(a, 1, 1);
}

// CIR:  cir.func @foo(%arg0: !cir.ptr<!void> loc({{.*}}))
// CIR:    [[PTR_ALLOC:%.*]] = cir.alloca !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>, ["a", init] {alignment = 8 : i64}
// CIR:    cir.store %arg0, [[PTR_ALLOC]] : !cir.ptr<!void>, cir.ptr <!cir.ptr<!void>>
// CIR:    [[PTR:%.*]] = cir.load [[PTR_ALLOC]] : cir.ptr <!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.prefetch([[PTR]] : !cir.ptr<!void>) locality(1) write
// CIR:    cir.return

// LLVM:  define void @foo(ptr [[ARG0:%.*]])
// LLVM:    [[PTR_ALLOC:%.*]] = alloca ptr, i64 1
// LLVM:    store ptr [[ARG0]], ptr [[PTR_ALLOC]]
// LLVM:    [[PTR:%.*]] = load ptr, ptr [[PTR_ALLOC]]
// LLVM:    call void @llvm.prefetch.p0(ptr [[PTR]], i32 1, i32 1, i32 1)
// LLVM:    ret void
