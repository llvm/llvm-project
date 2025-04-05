// Ensure that we can assemble VFPv4 by just specifying an armv7s target.

// REQUIRES: arm-registered-target
// RUN: %clang -c -target armv7s-apple-darwin -o /dev/null %s

vfma.f32 q1, q2, q3
