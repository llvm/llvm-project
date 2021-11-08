// RUN: %clang -S -emit-llvm -ffp-model=fast -emit-llvm %s -o - \
// RUN: | FileCheck %s \
// RUN: --check-prefixes=CHECK,CHECK-FAST

// RUN: %clang -S -emit-llvm -ffp-model=precise -emit-llvm %s -o - \
// RUN: | FileCheck %s \
// RUN: --check-prefixes=CHECK,CHECK-PRECISE

// RUN: %clang -S -emit-llvm -ffp-model=strict -emit-llvm %s -o - \
// RUN: | FileCheck %s \
// RUN: --check-prefixes=CHECK,CHECK-STRICT

// RUN: %clang -S -emit-llvm -ffp-model=strict -ffast-math -emit-llvm \
// RUN:  %s -o - | FileCheck %s \
// RUN: --check-prefixes CHECK,CHECK-STRICT-FAST

// RUN: %clang -S -emit-llvm -ffp-model=precise -ffast-math -emit-llvm \
// RUN: %s -o - | FileCheck %s \
// RUN: --check-prefixes CHECK,CHECK-FAST1

float mymuladd(float x, float y, float z) {
  // CHECK: define {{.*}} float @mymuladd(float noundef %x, float noundef %y, float noundef %z)
  return x * y + z;
  // CHECK-FAST: fmul fast float

  // CHECK-FAST: load float, float*
  // CHECK-FAST: fadd fast float

  // CHECK-PRECISE: load float, float*
  // CHECK-PRECISE: load float, float*
  // CHECK-PRECISE: load float, float*
  // CHECK-PRECISE: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})

  // CHECK-STRICT: load float, float*
  // CHECK-STRICT: load float, float*
  // CHECK-STRICT: call float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-STRICT: load float, float*
  // CHECK-STRICT: call float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}})

  // CHECK-STRICT-FAST: load float, float*
  // CHECK-STRICT-FAST: load float, float*
  // CHECK-STRICT-FAST: call fast float @llvm.experimental.constrained.fmul.f32(float {{.*}}, float {{.*}}, {{.*}})
  // CHECK-STRICT-FAST: load float, float*
  // CHECK-STRICT-FAST: call fast float @llvm.experimental.constrained.fadd.f32(float {{.*}}, float {{.*}}, {{.*}}

  // CHECK-FAST1: load float, float*
  // CHECK-FAST1: load float, float*
  // CHECK-FAST1: fmul fast float {{.*}}, {{.*}}
  // CHECK-FAST1: load float, float* {{.*}}
  // CHECK-FAST1: fadd fast float {{.*}}, {{.*}}
}
