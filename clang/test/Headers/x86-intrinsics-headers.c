// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -flax-vector-conversions=none %s
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -x c++ %s

// XFAIL: target=arm64ec-pc-windows-msvc
// These intrinsics are not yet implemented for Arm64EC.

#if defined(i386) || defined(__x86_64__)

#ifdef __SSE4_2__
// nmmintrin forwards to smmintrin.
#include <nmmintrin.h>
#endif

// immintrin includes all other intel intrinsic headers.
#include <immintrin.h>

#endif
