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

constexpr bool match_v16hi(__m256i _v, short a, short b, short c, short d, short e, short f, short g, short h, short i, short j, short k, short l, short m, short n, short o, short p) {
  __v16hi v = (__v16hi)_v;
  return v[0] == a && v[1] == b && v[2] == c && v[3] == d && v[4] == e && v[5] == f && v[6] == g && v[7] == h && v[8] == i && v[9] == j && v[10] == k && v[11] == l && v[12] == m && v[13] == n && v[14] == o && v[15] == p;
}

constexpr bool match_v32qi(__m256i _v, char __b00, char __b01, char __b02, char __b03, char __b04, char __b05, char __b06, char __b07,
                                       char __b08, char __b09, char __b10, char __b11, char __b12, char __b13, char __b14, char __b15,
                                       char __b16, char __b17, char __b18, char __b19, char __b20, char __b21, char __b22, char __b23,
                                       char __b24, char __b25, char __b26, char __b27, char __b28, char __b29, char __b30, char __b31) {
  __v32qi v = (__v32qi)_v;
  return v[ 0] == __b00 && v[ 1] == __b01 && v[ 2] == __b02 && v[ 3] == __b03 && v[ 4] == __b04 && v[ 5] == __b05 && v[ 6] == __b06 && v[ 7] ==  __b07 &&
         v[ 8] == __b08 && v[ 9] == __b09 && v[10] == __b10 && v[11] == __b11 && v[12] == __b12 && v[13] == __b13 && v[14] == __b14 && v[15] ==  __b15 &&
         v[16] == __b16 && v[17] == __b17 && v[18] == __b18 && v[19] == __b19 && v[20] == __b20 && v[21] == __b21 && v[22] == __b22 && v[23] ==  __b23 &&
         v[24] == __b24 && v[25] == __b25 && v[26] == __b26 && v[27] == __b27 && v[28] == __b28 && v[29] == __b29 && v[30] == __b30 && v[31] ==  __b31;
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

constexpr bool match_v32hi(__m512i _v, short __e00, short __e01, short __e02, short __e03, short __e04, short __e05, short __e06, short __e07,
                                       short __e08, short __e09, short __e10, short __e11, short __e12, short __e13, short __e14, short __e15,
                                       short __e16, short __e17, short __e18, short __e19, short __e20, short __e21, short __e22, short __e23,
                                       short __e24, short __e25, short __e26, short __e27, short __e28, short __e29, short __e30, short __e31) {
  __v32hi v = (__v32hi)_v;
  return v[ 0] == __e00 && v[ 1] == __e01 && v[ 2] == __e02 && v[ 3] == __e03 && v[ 4] == __e04 && v[ 5] == __e05 && v[ 6] == __e06 && v[ 7] ==  __e07 &&
         v[ 8] == __e08 && v[ 9] == __e09 && v[10] == __e10 && v[11] == __e11 && v[12] == __e12 && v[13] == __e13 && v[14] == __e14 && v[15] ==  __e15 &&
         v[16] == __e16 && v[17] == __e17 && v[18] == __e18 && v[19] == __e19 && v[20] == __e20 && v[21] == __e21 && v[22] == __e22 && v[23] ==  __e23 &&
         v[24] == __e24 && v[25] == __e25 && v[26] == __e26 && v[27] == __e27 && v[28] == __e28 && v[29] == __e29 && v[30] == __e30 && v[31] ==  __e31;
}

constexpr bool match_v64qi(__m512i _v, char __e00, char __e01, char __e02, char __e03, char __e04, char __e05, char __e06, char __e07,
                                       char __e08, char __e09, char __e10, char __e11, char __e12, char __e13, char __e14, char __e15,
                                       char __e16, char __e17, char __e18, char __e19, char __e20, char __e21, char __e22, char __e23,
                                       char __e24, char __e25, char __e26, char __e27, char __e28, char __e29, char __e30, char __e31,
                                       char __e32, char __e33, char __e34, char __e35, char __e36, char __e37, char __e38, char __e39,
                                       char __e40, char __e41, char __e42, char __e43, char __e44, char __e45, char __e46, char __e47,
                                       char __e48, char __e49, char __e50, char __e51, char __e52, char __e53, char __e54, char __e55,
                                       char __e56, char __e57, char __e58, char __e59, char __e60, char __e61, char __e62, char __e63) {
  __v64qi v = (__v64qi)_v;
  return v[ 0] == __e00 && v[ 1] == __e01 && v[ 2] == __e02 && v[ 3] == __e03 && v[ 4] == __e04 && v[ 5] == __e05 && v[ 6] == __e06 && v[ 7] == __e07 &&
         v[ 8] == __e08 && v[ 9] == __e09 && v[10] == __e10 && v[11] == __e11 && v[12] == __e12 && v[13] == __e13 && v[14] == __e14 && v[15] == __e15 &&
         v[16] == __e16 && v[17] == __e17 && v[18] == __e18 && v[19] == __e19 && v[20] == __e20 && v[21] == __e21 && v[22] == __e22 && v[23] == __e23 &&
         v[24] == __e24 && v[25] == __e25 && v[26] == __e26 && v[27] == __e27 && v[28] == __e28 && v[29] == __e29 && v[30] == __e30 && v[31] == __e31 &&
         v[32] == __e32 && v[33] == __e33 && v[34] == __e34 && v[35] == __e35 && v[36] == __e36 && v[37] == __e37 && v[38] == __e38 && v[39] == __e39 &&
         v[40] == __e40 && v[41] == __e41 && v[42] == __e42 && v[43] == __e43 && v[44] == __e44 && v[45] == __e45 && v[46] == __e46 && v[47] == __e47 &&
         v[48] == __e48 && v[49] == __e49 && v[50] == __e50 && v[51] == __e51 && v[52] == __e52 && v[53] == __e53 && v[54] == __e54 && v[55] == __e55 &&
         v[56] == __e56 && v[57] == __e57 && v[58] == __e58 && v[59] == __e59 && v[60] == __e60 && v[61] == __e61 && v[62] == __e62 && v[63] == __e63;
}

#define TEST_CONSTEXPR(...) static_assert(__VA_ARGS__)

#else

#define TEST_CONSTEXPR(...)

#endif
