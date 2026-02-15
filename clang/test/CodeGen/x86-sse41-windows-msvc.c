// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX

// This test verifies that SSE4.1 is enabled by default for Windows MSVC targets
// to support SIMD intrinsics like _mm_mullo_epi32

#include <immintrin.h>

__m128i test_sse41(void) {
  __m128i a = _mm_set1_epi32(5);
  __m128i b = _mm_set1_epi32(3);
  return _mm_mullo_epi32(a, b);
}

// CHECK: "target-features"="+sse4.1"
// LINUX-NOT: "target-features"="+sse4.1"
