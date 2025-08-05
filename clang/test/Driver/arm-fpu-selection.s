// REQUIRES: arm-registered-target
/// Ensures that when targeting an ARM target with an Asm file, clang
/// collects the features from the FPU. This is critical in the
/// activation of NEON for supported targets. The Cortex-R52 will be
/// used and tested for VFP and NEON Support

// RUN: %clang -target arm-none-eabi -mcpu=cortex-r52 -c %s -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-STDERR %s --allow-empty
// RUN: %clang -target arm-none-eabi -mcpu=cortex-r52 -c %s -o /dev/null -### 2>&1 | FileCheck --check-prefix=CHECK-TARGET-FEATURES %s

/// Check that no errors or warnings are present when assembling using cc1as.
// CHECK-STDERR-NOT: error:
// CHECK-STDERR-NOT: warning:

/// Check that NEON and VFPV5 have been activated when using Cortex-R52 when using cc1as
// CHECK-TARGET-FEATURES: "-target-feature" "+vfp2sp"
// CHECK-TARGET-FEATURES: "-target-feature" "+vfp3"
// CHECK-TARGET-FEATURES: "-target-feature" "+fp-armv8"
// CHECK-TARGET-FEATURES: "-target-feature" "+fp-armv8d16"
// CHECK-TARGET-FEATURES: "-target-feature" "+fp-armv8d16sp"
// CHECK-TARGET-FEATURES: "-target-feature" "+fp-armv8sp"
// CHECK-TARGET-FEATURES: "-target-feature" "+neon"

  vadd.f32 s0, s1, s2
  vadd.f64 d0, d1, d2
  vcvt.u32.f32 s0, s0, #1
  vcvt.u32.f64 d0, d0, #1
  vcvtb.f32.f16 s0, s1
  vcvtb.f64.f16 d0, s1
  vfma.f32 s0, s1, s2
  vfma.f64 d0, d1, d2
  vcvta.u32.f32 s0, s1
  vcvta.u32.f64 s0, d1
  vadd.f32 q0, q1, q2
  vcvt.f32.f16 q0, d1
  vfma.f32 q0, q1, q2
  vcvta.u32.f32 q0, q1
