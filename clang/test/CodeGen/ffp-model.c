// REQUIRES: x86-registered-target
// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=fast %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-FAST

// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=aggressive %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-AGGRESSIVE

// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=precise %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-PRECISE

// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=strict %s -o - \
// RUN: -target x86_64 | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT

// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=strict -ffast-math \
// RUN: -target x86_64 %s -o - | FileCheck %s \
// RUN: --check-prefixes CHECK,CHECK-STRICT-FAST

// RUN: %clang -S -emit-llvm -fenable-matrix -ffp-model=precise -ffast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-FAST1

float mymuladd(float x, float y, float z) {
  // CHECK: define{{.*}} float @mymuladd
  return x * y + z;

  // CHECK-AGGRESSIVE: fmul fast float
  // CHECK-AGGRESSIVE: load float, ptr
  // CHECK-AGGRESSIVE: fadd fast float

  // CHECK-FAST: fmul reassoc nsz arcp contract afn float
  // CHECK-FAST: load float, ptr
  // CHECK-FAST: fadd reassoc nsz arcp contract afn float

  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fmul fast float {{.*}}, {{.*}}
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fadd fast float {{.*}}, {{.*}}

  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: fmul fast float {{.*}}, {{.*}}
  // CHECK-FAST1: load float, ptr {{.*}}
  // CHECK-FAST1: fadd fast float {{.*}}, {{.*}}
}

typedef float __attribute__((ext_vector_type(2))) v2f;

void my_vec_muladd(v2f x, float y, v2f z, v2f *res) {
  // CHECK: define{{.*}}@my_vec_muladd
  *res = x * y + z;

  // CHECK-AGGRESSIVE: fmul fast <2 x float>
  // CHECK-AGGRESSIVE: load <2 x float>, ptr
  // CHECK-AGGRESSIVE: fadd fast <2 x float>

  // CHECK-FAST: fmul reassoc nsz arcp contract afn <2 x float>
  // CHECK-FAST: load <2 x float>, ptr
  // CHECK-FAST: fadd reassoc nsz arcp contract afn <2 x float>

  // CHECK-PRECISE: load <2 x float>, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load <2 x float>, ptr
  // CHECK-PRECISE: call <2 x float> @llvm.fmuladd.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, <2 x float> {{.*}})

  // CHECK-STRICT: load <2 x float>, ptr
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call <2 x float> @llvm.experimental.constrained.fmul.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, {{.*}})
  // CHECK-STRICT: load <2 x float>, ptr
  // CHECK-STRICT: call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load <2 x float>, ptr
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fmul fast <2 x float> {{.*}}, {{.*}}
  // CHECK-STRICT-FAST: load <2 x float>, ptr
  // CHECK-STRICT-FAST: fadd fast <2 x float> {{.*}}, {{.*}}

  // CHECK-FAST1: load <2 x float>, ptr
  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: fmul fast <2 x float> {{.*}}, {{.*}}
  // CHECK-FAST1: load <2 x float>, ptr {{.*}}
  // CHECK-FAST1: fadd fast <2 x float> {{.*}}, {{.*}}
}

typedef float __attribute__((matrix_type(2, 1))) m21f;

void my_m21_muladd(m21f x, float y, m21f z, m21f *res) {
  // CHECK: define{{.*}}@my_m21_muladd
  *res = x * y + z;

  // CHECK-AGGRESSIVE: fmul fast <2 x float>
  // CHECK-AGGRESSIVE: load <2 x float>, ptr
  // CHECK-AGGRESSIVE: fadd fast <2 x float>

  // CHECK-FAST: fmul reassoc nsz arcp contract afn <2 x float>
  // CHECK-FAST: load <2 x float>, ptr
  // CHECK-FAST: fadd reassoc nsz arcp contract afn <2 x float>

  // CHECK-PRECISE: load <2 x float>, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load <2 x float>, ptr
  // CHECK-PRECISE: call <2 x float> @llvm.fmuladd.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, <2 x float> {{.*}})

  // CHECK-STRICT: load <2 x float>, ptr
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call <2 x float> @llvm.experimental.constrained.fmul.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, {{.*}})
  // CHECK-STRICT: load <2 x float>, ptr
  // CHECK-STRICT: call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> {{.*}}, <2 x float> {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load <2 x float>, ptr
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fmul fast <2 x float> {{.*}}, {{.*}}
  // CHECK-STRICT-FAST: load <2 x float>, ptr
  // CHECK-STRICT-FAST: fadd fast <2 x float> {{.*}}, {{.*}}

  // CHECK-FAST1: load <2 x float>, ptr
  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: fmul fast <2 x float> {{.*}}, {{.*}}
  // CHECK-FAST1: load <2 x float>, ptr {{.*}}
  // CHECK-FAST1: fadd fast <2 x float> {{.*}}, {{.*}}
}

typedef float __attribute__((matrix_type(2, 2))) m22f;

void my_m22_muladd(m22f x, float y, m22f z, m22f *res) {
  // CHECK: define{{.*}}@my_m22_muladd
  *res = x * y + z;

  // CHECK-AGGRESSIVE: fmul fast <4 x float>
  // CHECK-AGGRESSIVE: load <4 x float>, ptr
  // CHECK-AGGRESSIVE: fadd fast <4 x float>

  // CHECK-FAST: fmul reassoc nsz arcp contract afn <4 x float>
  // CHECK-FAST: load <4 x float>, ptr
  // CHECK-FAST: fadd reassoc nsz arcp contract afn <4 x float>

  // CHECK-PRECISE: load <4 x float>, ptr
  // CHECK-PRECISE: load float, ptr
  // CHECK-PRECISE: load <4 x float>, ptr
  // CHECK-PRECISE: call <4 x float> @llvm.fmuladd.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}})

  // CHECK-STRICT: load <4 x float>, ptr
  // CHECK-STRICT: load float, ptr
  // CHECK-STRICT: call <4 x float> @llvm.experimental.constrained.fmul.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, {{.*}})
  // CHECK-STRICT: load <4 x float>, ptr
  // CHECK-STRICT: call <4 x float> @llvm.experimental.constrained.fadd.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load <4 x float>, ptr
  // CHECK-STRICT-FAST: load float, ptr
  // CHECK-STRICT-FAST: fmul fast <4 x float> {{.*}}, {{.*}}
  // CHECK-STRICT-FAST: load <4 x float>, ptr
  // CHECK-STRICT-FAST: fadd fast <4 x float> {{.*}}, {{.*}}

  // CHECK-FAST1: load <4 x float>, ptr
  // CHECK-FAST1: load float, ptr
  // CHECK-FAST1: fmul fast <4 x float> {{.*}}, {{.*}}
  // CHECK-FAST1: load <4 x float>, ptr {{.*}}
  // CHECK-FAST1: fadd fast <4 x float> {{.*}}, {{.*}}
}
