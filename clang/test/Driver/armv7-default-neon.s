// Ensure that we can assemble NEON by just specifying an armv7
// Apple or Windows target.

// REQUIRES: arm-registered-target
// RUN: %clang -c -target armv7-apple-darwin -o /dev/null %s
// RUN: %clang -c -target armv7-windows -o /dev/null %s

vadd.i32 q0, q0, q0
