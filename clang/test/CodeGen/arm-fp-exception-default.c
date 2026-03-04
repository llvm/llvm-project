// REQUIRES: arm-registered-target
//
// Test that -ffp-exception-behavior=maytrap is accepted on ARM32 and
// produces constrained FP intrinsics.  HasStrictFP is set to true in
// ARMTargetInfo so the frontend no longer rejects this flag on ARM.
// The STRICT_F* nodes are handled via the mutateStrictFPToFP expansion path
// in the backend (see fp-maytrap-default.ll for the codegen half).

// RUN: %clang_cc1 -triple armv7a-none-eabi -target-cpu cortex-a9 \
// RUN:     -ffp-exception-behavior=maytrap \
// RUN:     -disable-O0-optnone -emit-llvm %s -o - \
// RUN:     | FileCheck -check-prefix=IR %s

float guarded_div(float a, float b) {
  // IR-LABEL: define {{.*}} @guarded_div(
  // IR: call float @llvm.experimental.constrained.fdiv.f32(
  // IR-SAME: metadata !"fpexcept.maytrap"
  // IR: attributes {{.*}} = { {{.*}}strictfp{{.*}} }
  return a / b;
}
