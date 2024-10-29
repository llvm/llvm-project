/* Helper methods for builtin intrinsic tests */

#include <immintrin.h>

#if defined(__cplusplus) && (__cplusplus >= 201103L)

constexpr bool match_m128(__m128 v, float a, float b, float c, float d) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_m128d(__m128d v, double a, double b) {
  return v[0] == a && v[1] == b;
}

constexpr bool match_m128i(__m128i v, unsigned long long a, unsigned long long b) {
  return v[0] == a && v[1] == b;
}

constexpr bool match_m256(__m256 v, float a, float b, float c, float d, float e, float f, float g, float h) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m256d(__m256d v, double a, double b, double c, double d) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_m256i(__m256i v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_m512(__m512 v, float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l, float m, float n, float o, float p) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h && v[8] == i && v[9] == j && v[10] == k && v[11] == l && v[12] == m && v[13] == n && v[14] == o && v[15] == p;
}

constexpr bool match_m512d(__m512d v, double a, double b, double c, double d, double e, double f, double g, double h) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m512i(__m512i v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long g, unsigned long long h) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

#define TEST_CONSTEXPR(...) static_assert(__VA_ARGS__)

#else

#define TEST_CONSTEXPR(...)

#endif
