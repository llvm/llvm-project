/* Helper methods for builtin intrinsic tests */

#include <immintrin.h>

#if defined(__cplusplus) && (__cplusplus >= 201103L)

constexpr bool match_m64(__m64 _v, unsigned long long a) {
  __v1du v = (__v1du)_v;
  return v[0] == a;
}

constexpr bool match_v1di(__m64 v, long long a) {
  return v[0] == a;
}

constexpr bool match_v2si(__m64 _v, int a, int b) {
  __v2si v = (__v2si)_v;
  return v[0] == a && v[1] == b;
}

constexpr bool match_v4hi(__m64 _v, short a, short b, short c, short d) {
  __v4hi v = (__v4hi)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_v8qi(__m64 _v, char a, char b, char c, char d, char e, char f, char g, char h) {
  __v8qi v = (__v8qi)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m128(__m128 v, float a, float b, float c, float d) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_m128d(__m128d v, double a, double b) {
  return v[0] == a && v[1] == b;
}

constexpr bool match_m128i(__m128i _v, unsigned long long a, unsigned long long b) {
  __v2du v = (__v2du)_v;
  return v[0] == a && v[1] == b;
}

constexpr bool match_v2di(__m128i v, long long a, long long b) {
  return v[0] == a && v[1] == b;
}

constexpr bool match_v4si(__m128i _v, int a, int b, int c, int d) {
  __v4si v = (__v4si)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_v8hi(__m128i _v, short a, short b, short c, short d, short e, short f, short g, short h) {
  __v8hi v = (__v8hi)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_v16qi(__m128i _v, char a, char b, char c, char d, char e, char f, char g, char h, char i, char j, char k, char l, char m, char n, char o, char p) {
  __v16qi v = (__v16qi)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h && v[8] == i && v[9] == j && v[10] == k && v[11] == l && v[12] == m && v[13] == n && v[14] == o && v[15] == p;
}

constexpr bool match_m256(__m256 v, float a, float b, float c, float d, float e, float f, float g, float h) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m256d(__m256d v, double a, double b, double c, double d) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_m256i(__m256i _v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d) {
  __v4du v = (__v4du)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_v4di(__m256i _v, long long a, long long b, long long c, long long d) {
  __v4di v = (__v4di)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d;
}

constexpr bool match_v8si(__m256i _v, int a, int b, int c, int d, int e, int f, int g, int h) {
  __v8si v = (__v8si)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m512(__m512 v, float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l, float m, float n, float o, float p) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h && v[8] == i && v[9] == j && v[10] == k && v[11] == l && v[12] == m && v[13] == n && v[14] == o && v[15] == p;
}

constexpr bool match_m512d(__m512d v, double a, double b, double c, double d, double e, double f, double g, double h) {
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_m512i(__m512i _v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long g, unsigned long long h) {
  __v8du v = (__v8du)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_v8di(__m512i _v, long long a, long long b, long long c, long long d, long long e, long long f, long long g, long long h) {
  __v8di v = (__v8di)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h;
}

constexpr bool match_v16si(__m512i _v, int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o, int p) {
  __v16si v = (__v16si)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h && v[8] == i && v[9] == j && v[10] == k && v[11] == l && v[12] == m && v[13] == n && v[14] == o && v[15] == p;
}

#define TEST_CONSTEXPR(...) static_assert(__VA_ARGS__)

#else

#define TEST_CONSTEXPR(...)

#endif
