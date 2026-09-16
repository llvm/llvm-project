// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// On AArch64 __fp16 may be passed and returned directly (the type is not
// legalized to i16). Even though the value crosses the ABI as a half,
// arithmetic on __fp16 is still performed by promoting to float and
// truncating back, whereas _Float16 uses native half arithmetic.

__fp16 ret_half(__fp16 a) { return a; }

// CIR-LABEL: cir.func {{.*}}@ret_half(%arg0: !cir.f16 {{.*}}) -> !cir.f16
// LLVM-LABEL: define {{.*}}half @ret_half(half {{.*}}%{{.*}})

// __fp16: storage-only semantics, arithmetic promoted through float.
__fp16 add_halves(__fp16 a, __fp16 b) { return a + b; }

// CIR-LABEL: cir.func {{.*}}@add_halves(%arg0: !cir.f16 {{.*}}, %arg1: !cir.f16 {{.*}}) -> !cir.f16
// CIR:   %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[AF:.*]] = cir.cast floating %[[A]] : !cir.f16 -> !cir.float
// CIR:   %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[BF:.*]] = cir.cast floating %[[B]] : !cir.f16 -> !cir.float
// CIR:   %[[SUM:.*]] = cir.fadd %[[AF]], %[[BF]] : !cir.float
// CIR:   %{{.*}} = cir.cast floating %[[SUM]] : !cir.float -> !cir.f16

// LLVM-LABEL: define {{.*}}half @add_halves(half {{.*}}%{{.*}}, half {{.*}}%{{.*}})
// LLVM:   %[[EA:.*]] = fpext half %{{.*}} to float
// LLVM:   %[[EB:.*]] = fpext half %{{.*}} to float
// LLVM:   %[[ADD:.*]] = fadd float %[[EA]], %[[EB]]
// LLVM:   %{{.*}} = fptrunc float %[[ADD]] to half

// _Float16: native half arithmetic, no promotion to float.
_Float16 native_add(_Float16 a, _Float16 b) { return a + b; }

// CIR-LABEL: cir.func {{.*}}@native_add(%arg0: !cir.f16 {{.*}}, %arg1: !cir.f16 {{.*}}) -> !cir.f16
// CIR:   %[[A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %[[B:.*]] = cir.load {{.*}} : !cir.ptr<!cir.f16>, !cir.f16
// CIR:   %{{.*}} = cir.fadd %[[A]], %[[B]] : !cir.f16
// CIR-NOT: cir.cast floating {{.*}}!cir.f16 -> !cir.float

// LLVM-LABEL: define {{.*}}half @native_add(half {{.*}}%{{.*}}, half {{.*}}%{{.*}})
// LLVM:   %{{.*}} = fadd half %{{.*}}, %{{.*}}
// LLVM-NOT: fpext half
