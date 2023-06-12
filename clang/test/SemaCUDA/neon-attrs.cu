// CPU-side compilation on ARM with neon enabled (no errors expected).
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -aux-triple nvptx64 -x cuda -fsyntax-only -verify=quiet %s

// CPU-side compilation on ARM with neon disabled.
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature -neon -aux-triple nvptx64 -x cuda -fsyntax-only -verify %s

// GPU-side compilation on ARM (no errors expected).
// RUN: %clang_cc1 -triple nvptx64 -aux-triple arm64-linux-gnu -fcuda-is-device -x cuda -fsyntax-only -verify=quiet %s

// Regular C++ compilation on ARM with neon enabled (no errors expected).
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -x c++ -fsyntax-only -verify=quiet %s

// Regular C++ compilation on ARM with neon disabled.
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature -neon -x c++ -fsyntax-only -verify %s

// quiet-no-diagnostics
typedef __attribute__((neon_vector_type(4))) float float32x4_t;
// expected-error@-1 {{'neon_vector_type' attribute is not supported on targets missing 'neon' or 'mve'}}
typedef unsigned char poly8_t;
typedef __attribute__((neon_polyvector_type(8))) poly8_t poly8x8_t;
// expected-error@-1 {{'neon_polyvector_type' attribute is not supported on targets missing 'neon' or 'mve'}}
