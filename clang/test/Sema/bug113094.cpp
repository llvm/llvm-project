// RUN: %clang -fopenmp --offload-arch=sm_90 -nocudalib -target aarch64-unknown-linux-gnu -c -Xclang -verify %s
// REQUIRES: aarch64-registered-target

// expected-no-diagnostics

typedef __attribute__ ((__neon_vector_type__ (4))) float __f32x4_t;
