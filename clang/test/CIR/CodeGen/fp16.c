// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// On x86_64 __fp16 is a storage-only type: it is stored as a half, but all
// arithmetic is performed by promoting to float and truncating back. Verify
// that ClangIR emits the half storage type directly and routes conversions
// through float (as opposed to the removed llvm.convert.to/from.fp16
// intrinsics).

__fp16 g;
__fp16 gi = 1.5;

// CIR-DAG: cir.global external @g = #cir.fp<0.000000e+00> : !cir.f16
// CIR-DAG: cir.global external @gi = #cir.fp<1.500000e+00> : !cir.f16

// LLVM-DAG: @g = global half 0.000000e+00
// LLVM-DAG: @gi = global half 1.500000e+00

float load_to_float(__fp16 *p) { return *p; }

// CIR-LABEL: cir.func {{.*}}@load_to_float(%arg0: !cir.ptr<!cir.f16> {{.*}}) -> !cir.float
// CIR:   %[[VAL:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %{{.*}} = cir.cast floating %[[VAL]] : !cir.f16 -> !cir.float

// LLVM-LABEL: define {{.*}}float @load_to_float(ptr {{.*}}%{{.*}})
// LLVM:   %[[LD:.*]] = load half, ptr %{{.*}}
// LLVM:   %{{.*}} = fpext half %[[LD]] to float

void store_from_float(__fp16 *p, float f) { *p = f; }

// CIR-LABEL: cir.func {{.*}}@store_from_float(%arg0: !cir.ptr<!cir.f16> {{.*}}, %arg1: !cir.float {{.*}})
// CIR:   %[[F:.*]] = cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[H:.*]] = cir.cast floating %[[F]] : !cir.float -> !cir.f16
// CIR:   cir.store {{.*}} %[[H]], %{{.*}} : !cir.f16, !cir.ptr<!cir.f16>

// LLVM-LABEL: define {{.*}}void @store_from_float(ptr {{.*}}%{{.*}}, float {{.*}}%{{.*}})
// LLVM:   %[[TR:.*]] = fptrunc float %{{.*}} to half
// LLVM:   store half %[[TR]], ptr %{{.*}}

int half_to_int(__fp16 *p) { return (int)*p; }

// CIR-LABEL: cir.func {{.*}}@half_to_int(%arg0: !cir.ptr<!cir.f16> {{.*}}) -> !s32i
// CIR:   %[[VAL:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[FP:.*]] = cir.cast floating %[[VAL]] : !cir.f16 -> !cir.float
// CIR:   %{{.*}} = cir.cast float_to_int %[[FP]] : !cir.float -> !s32i

// LLVM-LABEL: define {{.*}}i32 @half_to_int(ptr {{.*}}%{{.*}})
// LLVM:   %[[LD:.*]] = load half, ptr %{{.*}}
// LLVM:   %[[EXT:.*]] = fpext half %[[LD]] to float
// LLVM:   %{{.*}} = fptosi float %[[EXT]] to i32

void add_halves(__fp16 *r, __fp16 *a, __fp16 *b) { *r = *a + *b; }

// CIR-LABEL: cir.func {{.*}}@add_halves(%arg0: !cir.ptr<!cir.f16> {{.*}}, %arg1: !cir.ptr<!cir.f16> {{.*}}, %arg2: !cir.ptr<!cir.f16> {{.*}})
// CIR:   %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[AF:.*]] = cir.cast floating %[[A]] : !cir.f16 -> !cir.float
// CIR:   %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[BF:.*]] = cir.cast floating %[[B]] : !cir.f16 -> !cir.float
// CIR:   %[[SUM:.*]] = cir.fadd %[[AF]], %[[BF]] : !cir.float
// CIR:   %{{.*}} = cir.cast floating %[[SUM]] : !cir.float -> !cir.f16

// LLVM-LABEL: define {{.*}}void @add_halves(ptr {{.*}}%{{.*}}, ptr {{.*}}%{{.*}}, ptr {{.*}}%{{.*}})
// LLVM:   %[[EA:.*]] = fpext half %{{.*}} to float
// LLVM:   %[[EB:.*]] = fpext half %{{.*}} to float
// LLVM:   %[[ADD:.*]] = fadd float %[[EA]], %[[EB]]
// LLVM:   %{{.*}} = fptrunc float %[[ADD]] to half
