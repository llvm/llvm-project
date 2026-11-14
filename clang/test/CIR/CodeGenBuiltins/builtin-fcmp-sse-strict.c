// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-exception-behavior=strict -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));

// CIR-LABEL: @test_cmpeqps
// LLVM-LABEL: @test_cmpeqps
__m128 test_cmpeqps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(oeq, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpeqps(A, B);
}

// CIR-LABEL: @test_cmpltps
// LLVM-LABEL: @test_cmpltps
__m128 test_cmpltps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(olt, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpltps(A, B);
}

// CIR-LABEL: @test_cmpleps
// LLVM-LABEL: @test_cmpleps
__m128 test_cmpleps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(ole, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpleps(A, B);
}

// CIR-LABEL: @test_cmpunordps
// LLVM-LABEL: @test_cmpunordps
__m128 test_cmpunordps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(uno, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpunordps(A, B);
}

// CIR-LABEL: @test_cmpneqps
// LLVM-LABEL: @test_cmpneqps
__m128 test_cmpneqps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(une, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"une", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpneqps(A, B);
}

// CIR-LABEL: @test_cmpnltps
// LLVM-LABEL: @test_cmpnltps
__m128 test_cmpnltps(__m128 A, __m128 B) {
  // CIR: cir.vec.fcmp(uge, {{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> signaling fenv<{{.*}}strict_except = true>
  // LLVM: call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpnltps(A, B);
}

// CIR-LABEL: @test_cmpnlepd
// LLVM-LABEL: @test_cmpnlepd
__m128d test_cmpnlepd(__m128d A, __m128d B) {
  // CIR: cir.vec.fcmp(ugt, {{.*}}) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i> signaling fenv<{{.*}}strict_except = true>
  // LLVM: call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> {{.*}}, <2 x double> {{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpnlepd(A, B);
}

// CIR-LABEL: @test_cmpordpd
// LLVM-LABEL: @test_cmpordpd
__m128d test_cmpordpd(__m128d A, __m128d B) {
  // CIR: cir.vec.fcmp(ord, {{.*}}) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i> fenv<{{.*}}strict_except = true>
  // LLVM: call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> {{.*}}, <2 x double> {{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  return __builtin_ia32_cmpordpd(A, B);
}
