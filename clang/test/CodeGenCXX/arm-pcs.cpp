// Covers a bug fix for ABI selection with homogenous aggregates:
//  See: https://bugs.llvm.org/show_bug.cgi?id=39982

// REQUIRES: arm-registered-target
// RUN: %clang -mfloat-abi=hard --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=HARD,CHECK
// RUN: %clang -mfloat-abi=softfp --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=SOFTFP,CHECK
// RUN: %clang -mfloat-abi=soft --target=armv7-unknown-linux-gnueabi -O3 -S -o - %s | FileCheck %s -check-prefixes=SOFT,CHECK

// aapcs-vfp is only supported when fpregs is available.
#ifdef __ARM_FP
#define PCS_VFP __attribute__((pcs("aapcs-vfp")))
#else
#define PCS_VFP
#endif

struct S {
  float f;
  float d;
  float c;
  float t;
};

// Variadic functions should always marshal for the base standard.
// See section 5.5 (Parameter Passing) of the AAPCS.
float PCS_VFP variadic(S s, ...) {
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

float PCS_VFP baz(float x, float y) {
  // CHECK-NOT: mov s{{[0-9]+}}, r{{[0-9]+}}
  // SOFT: mov r{{[0-9]+}}, r{{[0-9]+}}
  // SOFTFP: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  // HARD: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
  return y;
}

float PCS_VFP foo(S s) {
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
