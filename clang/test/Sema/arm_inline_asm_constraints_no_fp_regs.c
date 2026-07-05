// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple arm -target-feature -fpregs -verify=arm-nofp %s

// w: A 32, 64, or 128-bit floating-point/SIMD register: s0-s31, d0-d31, or q0-q15.
float test_w(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=w"(x)
          : "w"(x)); // No error expected.
  // arm-nofp-error@7 {{invalid output constraint '=w' in asm}}
  return x;
}

// x: A 32, 64, or 128-bit floating-point/SIMD register: s0-s15, d0-d7, or q0-q3.
float test_x(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=x"(x)
          : "x"(x)); // No error expected.
  // arm-nofp-error@16 {{invalid output constraint '=x' in asm}}
  return x;
}

// t: A 32, 64, or 128-bit floating-point/SIMD register: s0-s31, d0-d15, or q0-q7.
float test_t(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=t"(x)
          : "t"(x)); // No error expected.
  // arm-nofp-error@25 {{invalid output constraint '=t' in asm}}
  return x;
}
