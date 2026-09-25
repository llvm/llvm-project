// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -menable-no-nans %s -o - | FileCheck %s --check-prefix=CIR-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -menable-no-nans %s -o - | FileCheck %s --check-prefix=LLVM-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -menable-no-nans %s -o - | FileCheck %s --check-prefix=LLVM-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -ffast-math -ffp-contract=fast %s -o - | FileCheck %s --check-prefix=CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -ffast-math -ffp-contract=fast %s -o - | FileCheck %s --check-prefix=LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -ffast-math -ffp-contract=fast %s -o - | FileCheck %s --check-prefix=LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -mreassociate %s -o - | FileCheck %s --check-prefix=CIR-PRAGMA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -mreassociate %s -o - | FileCheck %s --check-prefix=LLVM-PRAGMA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:   -mreassociate %s -o - | FileCheck %s --check-prefix=LLVM-PRAGMA

typedef float v4sf __attribute__((vector_size(16)));
typedef int v4si __attribute__((vector_size(16)));

float test_assoc(v4sf x, float start) {
  // CIR-NNAN-LABEL: @test_assoc
  // CIR-NNAN: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan, reassoc>}
  // CIR-FAST-LABEL: @test_assoc
  // CIR-FAST: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<fast>}
  // LLVM-NNAN-LABEL: @test_assoc
  // LLVM-NNAN: call reassoc nnan float @llvm.vector.reduce.fadd.v4f32(
  // LLVM-FAST-LABEL: @test_assoc
  // LLVM-FAST: call fast float @llvm.vector.reduce.fadd.v4f32(
  return __builtin_reduce_assoc_fadd(x, start);
}

float test_in_order(v4sf x, float start) {
  // CIR-NNAN-LABEL: @test_in_order
  // CIR-NNAN: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan>}
  // CIR-FAST-LABEL: @test_in_order
  // CIR-FAST: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<fast>}
  // LLVM-NNAN-LABEL: @test_in_order
  // LLVM-NNAN: call nnan float @llvm.vector.reduce.fadd.v4f32(
  // LLVM-FAST-LABEL: @test_in_order
  // LLVM-FAST: call fast float @llvm.vector.reduce.fadd.v4f32(
  return __builtin_reduce_in_order_fadd(x, start);
}

float test_max(v4sf x) {
  // CIR-NNAN-LABEL: @test_max
  // CIR-NNAN: cir.vec.reduce(fmax, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan>}
  // CIR-FAST-LABEL: @test_max
  // CIR-FAST: cir.vec.reduce(fmax, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<fast>}
  // LLVM-NNAN-LABEL: @test_max
  // LLVM-NNAN: call nnan float @llvm.vector.reduce.fmax.v4f32(
  // LLVM-FAST-LABEL: @test_max
  // LLVM-FAST: call fast float @llvm.vector.reduce.fmax.v4f32(
  return __builtin_reduce_max(x);
}

float test_min(v4sf x) {
  // CIR-NNAN-LABEL: @test_min
  // CIR-NNAN: cir.vec.reduce(fmin, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<nnan>}
  // CIR-FAST-LABEL: @test_min
  // CIR-FAST: cir.vec.reduce(fmin, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<fast>}
  // LLVM-NNAN-LABEL: @test_min
  // LLVM-NNAN: call nnan float @llvm.vector.reduce.fmin.v4f32(
  // LLVM-FAST-LABEL: @test_min
  // LLVM-FAST: call fast float @llvm.vector.reduce.fmin.v4f32(
  return __builtin_reduce_min(x);
}

int test_int_max(v4si x) {
  // CIR-NNAN-LABEL: @test_int_max
  // CIR-NNAN: cir.vec.reduce(smax, {{.*}}) : (!cir.vector<4 x !s32i>) -> !s32i{{( loc.*)?$}}
  // CIR-FAST-LABEL: @test_int_max
  // CIR-FAST: cir.vec.reduce(smax, {{.*}}) : (!cir.vector<4 x !s32i>) -> !s32i{{( loc.*)?$}}
  // LLVM-NNAN-LABEL: @test_int_max
  // LLVM-NNAN: call i32 @llvm.vector.reduce.smax.v4i32(
  // LLVM-FAST-LABEL: @test_int_max
  // LLVM-FAST: call i32 @llvm.vector.reduce.smax.v4i32(
  return __builtin_reduce_max(x);
}

float test_pragma_reassociate(v4sf x, float start) {
#pragma clang fp reassociate(on)
  // CIR-PRAGMA-LABEL: @test_pragma_reassociate
  // CIR-PRAGMA: cir.vec.reduce(fadd, {{.*}}) {{.*}} {fastmath_flags = #cir.fastmath<reassoc>}
  // LLVM-PRAGMA-LABEL: @test_pragma_reassociate
  // LLVM-PRAGMA: call reassoc float @llvm.vector.reduce.fadd.v4f32(
  return __builtin_reduce_in_order_fadd(x, start);
}

float test_pragma_no_reassociate(v4sf x, float start) {
#pragma clang fp reassociate(off)
  // CIR-PRAGMA-LABEL: @test_pragma_no_reassociate
  // CIR-PRAGMA: cir.vec.reduce(fadd, {{.*}}) : (!cir.vector<4 x !cir.float>, !cir.float) -> !cir.float{{( loc.*)?$}}
  // LLVM-PRAGMA-LABEL: @test_pragma_no_reassociate
  // LLVM-PRAGMA: call float @llvm.vector.reduce.fadd.v4f32(
  return __builtin_reduce_in_order_fadd(x, start);
}
