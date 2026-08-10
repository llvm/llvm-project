// RUN: %check_clang_tidy -std=c++11-or-later %s portability-simd-intrinsics %t -- \
// RUN:  -config='{CheckOptions: { \
// RUN:    portability-simd-intrinsics.Suggest: false \
// RUN:  }}' -- -target x86_64
// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s portability-simd-intrinsics -check-suffix=BEFORE-CXX20 %t -- \
// RUN:  -config='{CheckOptions: { \
// RUN:    portability-simd-intrinsics.Suggest: true \
// RUN:  }}' -- -target x86_64
// RUN: %check_clang_tidy -std=c++20-or-later %s portability-simd-intrinsics -check-suffix=CXX20 %t -- \
// RUN:  -config='{CheckOptions: { \
// RUN:    portability-simd-intrinsics.Suggest: true \
// RUN:  }}' -- -target x86_64

typedef long long __m128i __attribute__((vector_size(16)));
typedef double __m256 __attribute__((vector_size(32)));

__m128i _mm_add_epi32(__m128i, __m128i);
__m256 _mm256_load_pd(double const *);
void _mm256_store_pd(double *, __m256);

int _mm_add_fake(int, int);

void X86() {
  __m128i i0, i1;
  __m256 d0;

  _mm_add_epi32(i0, i1);
  // CHECK-MESSAGES-BEFORE-CXX20: :[[@LINE-1]]:3: warning: '_mm_add_epi32' can be replaced by operator+ on std::experimental::simd objects [portability-simd-intrinsics]
  // CHECK-MESSAGES-CXX20: :[[@LINE-2]]:3: warning: '_mm_add_epi32' can be replaced by operator+ on std::simd objects [portability-simd-intrinsics]
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: '_mm_add_epi32' is a non-portable x86_64 intrinsic function [portability-simd-intrinsics]
  d0 = _mm256_load_pd(0);
  _mm256_store_pd(0, d0);

  _mm_add_fake(0, 1);
}
