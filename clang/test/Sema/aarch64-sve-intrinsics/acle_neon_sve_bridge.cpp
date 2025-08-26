// RUN: %clang_cc1 -triple aarch64-none-linux-gnu  -verify -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

// Test that we can use __ARM_NEON_SVE_BRIDGE to guard to inclusion of arm_neon_sve_bridge.h,
// and use the associated intrinsics via a target() attribute.

// expected-no-diagnostics

#ifdef __ARM_NEON_SVE_BRIDGE
#include <arm_neon_sve_bridge.h>
#endif

uint32x4_t __attribute__((target("+sve"))) foo(svuint32_t a) {
    return svget_neonq_u32(a);
}