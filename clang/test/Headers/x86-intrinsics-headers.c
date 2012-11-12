// RUN: %clang -fsyntax-only -ffreestanding %s
// RUN: %clang -fsyntax-only -ffreestanding -fno-lax-vector-conversions %s
// RUN: %clangxx -fsyntax-only -ffreestanding -x c++ %s

#if defined(i386) || defined(__x86_64__)

#ifdef __SSE4_2__
// nmmintrin forwards to smmintrin.
#include <nmmintrin.h>
#endif

// immintrin includes all other intel intrinsic headers.
#include <immintrin.h>

#endif
