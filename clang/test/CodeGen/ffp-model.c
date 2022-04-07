// REQUIRES: x86-registered-target
// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -ffp-model=fast -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-FAST

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -ffp-model=precise %s -o - \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-PRECISE

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -ffp-model=strict %s -o - \
// RUN: -target x86_64 | FileCheck %s --check-prefixes=CHECK,CHECK-STRICT

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -ffp-model=strict -ffast-math \
// RUN: -target x86_64 %s -o - | FileCheck %s \
// RUN: --check-prefixes CHECK,CHECK-STRICT-FAST

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -ffp-model=precise -ffast-math \
// RUN: %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-FAST1

float mymuladd(float x, float y, float z) {
  // CHECK: define{{.*}} float @mymuladd
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
