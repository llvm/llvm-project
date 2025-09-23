// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

#include <immintrin.h> // AVX/AVX512 헤더

// // 테스트하려는 AVX/AVX512 내장 함수를 사용하는 constexpr 함수
// constexpr int test_avx_subvector_extraction() {
//   __m256i a = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

//   // 이슈의 핵심: 이 내장 함수 호출이 constexpr 문맥에서 가능해야 함
//   __m128i sub = _mm256_extracti128_si256(a, 0);

//   return _mm_cvtsi128_si32(sub); // 결과를 int로 변환하여 리턴
// }

// // 이 상수는 컴파일 시간에 평가되어야 함
// constexpr int result = test_avx_subvector_extraction();

// static_assert(result == 0, "Incorrect result");

#include <immintrin.h>

constexpr __m128 test(__m256 a) {
  return _mm256_extractf128_ps(a, 1);
}