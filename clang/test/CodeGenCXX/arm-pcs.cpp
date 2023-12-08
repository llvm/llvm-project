// Covers a bug fix for ABI selection with homogenous aggregates:
//  See: https://bugs.llvm.org/show_bug.cgi?id=39982

// REQUIRES: arm-registered-target
// RUN: %clang -mfloat-abi=hard --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=HARD,CHECK
// RUN: %clang -mfloat-abi=softfp --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=SOFTFP,CHECK
// RUN: %clang -mfloat-abi=soft --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=SOFT,CHECK

struct S {
  float f;
  float d;
  float c;
  float t;
};

// Variadic functions should always marshal for the base standard.
// See section 5.5 (Parameter Passing) of the AAPCS.
float __attribute__((pcs("aapcs-vfp"))) variadic(S s, ...) {
  // CHECK-NOT: vmov s{{[0-9]+}}, s{{[0-9]+}}
  // CHECK: mov r{{[0-9]+}}, r{{[0-9]+}}
  return s.d;
}

float no_attribute(S s) {
  // SOFT: mov r{{[0-9]+}}, r{{[0-9]+}}
  // SOFTFP: mov r{{[0-9]+}}, r{{[0-9]+}}
  // HARD: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  return s.d;
}

float __attribute__((pcs("aapcs-vfp"))) baz(float x, float y) {
  // CHECK-NOT: mov s{{[0-9]+}}, r{{[0-9]+}}
  // SOFT: mov r{{[0-9]+}}, r{{[0-9]+}}
  // SOFTFP: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  // HARD: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  return y;
}

float __attribute__((pcs("aapcs-vfp"))) foo(S s) {
  // CHECK-NOT: mov s{{[0-9]+}}, r{{[0-9]+}}
  // SOFT: mov r{{[0-9]+}}, r{{[0-9]+}}
  // SOFTFP: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  // HARD: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  return s.d;
}

float __attribute__((pcs("aapcs"))) bar(S s) {
  // CHECK-NOT: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  // CHECK: mov r{{[0-9]+}}, r{{[0-9]+}}
  return s.d;
}
