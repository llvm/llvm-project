/* Helper methods for builtin intrinsic tests */

#include <immintrin.h>

#if defined(__cplusplus) && (__cplusplus >= 201103L)

constexpr bool match_m128(__m128 v, float x, float y, float z, float w) {
  return v[0] == x && v[1] == y && v[2] == z && v[3] == w;
}

constexpr bool match_m128d(__m128d v, double x, double y) {
  return v[0] == x && v[1] == y;
}

constexpr bool match_m128i(__m128i v, unsigned long long x, unsigned long long y) {
  return v[0] == x && v[1] == y;
}

#define TEST_CONSTEXPR(...) static_assert(__VA_ARGS__)

#else

#define TEST_CONSTEXPR(...)

#endif