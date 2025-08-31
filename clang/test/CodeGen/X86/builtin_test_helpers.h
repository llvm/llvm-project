/* Helper methods for builtin intrinsic tests */

#include <immintrin.h>

#if defined(__cplusplus) && (__cplusplus >= 201103L)


#define TEST_CONSTEXPR(...) static_assert(__VA_ARGS__, #__VA_ARGS__)

// alL constexpr functions C++11.


constexpr bool match_m64(__m64 _v, unsigned long long a) {
  return ((__v1du)_v)[0] == a;
}

constexpr bool match_v1di(__m64 v, long long a) {
  return v[0] == a;
}

constexpr bool match_v1du(__m64 _v, unsigned long long a) {
  return ((__v1du)_v)[0] == a;
}

constexpr bool match_v2si(__m64 _v, int a, int b) {
  return ((__v2si)_v)[0] == a && ((__v2si)_v)[1] == b;
}

constexpr bool match_v2su(__m64 _v, unsigned a, unsigned b) {
  return ((__v2su)_v)[0] == a && ((__v2su)_v)[1] == b;
}

constexpr bool match_v4hi(__m64 _v, short a, short b, short c, short d) {
  return ((__v4hi)_v)[0] == a && ((__v4hi)_v)[1] == b && ((__v4hi)_v)[2] == c && ((__v4hi)_v)[3] == d;
}

constexpr bool match_v4hu(__m64 _v, unsigned short a, unsigned short b, unsigned short c, unsigned short d) {
  return ((__v4hu)_v)[0] == a && ((__v4hu)_v)[1] == b && ((__v4hu)_v)[2] == c && ((__v4hu)_v)[3] == d;
}

constexpr bool match_v8qi(__m64 _v, signed char a, signed char b, signed char c, signed char d, signed char e, signed char f, signed char g, signed char h) {
  return ((__v8qs)_v)[0] == a && ((__v8qs)_v)[1] == b && ((__v8qs)_v)[2] == c && ((__v8qs)_v)[3] == d && ((__v8qs)_v)[4] == e && ((__v8qs)_v)[5] == f && ((__v8qs)_v)[6] == g && ((__v8qs)_v)[7] == h;
}

constexpr bool match_v8qu(__m64 _v, unsigned char a, unsigned char b, unsigned char c, unsigned char d, unsigned char e, unsigned char f, unsigned char g, unsigned char h) {
  return ((__v8qu)_v)[0] == a && ((__v8qu)_v)[1] == b && ((__v8qu)_v)[2] == c && ((__v8qu)_v)[3] == d && ((__v8qu)_v)[4] == e && ((__v8qu)_v)[5] == f && ((__v8qu)_v)[6] == g && ((__v8qu)_v)[7] == h;
}

constexpr bool match_m128(__m128 _v, float a, float b, float c, float d) {
  return ((__v4su)_v)[0] == __builtin_bit_cast(unsigned, a) && ((__v4su)_v)[1] == __builtin_bit_cast(unsigned, b) && ((__v4su)_v)[2] == __builtin_bit_cast(unsigned, c) && ((__v4su)_v)[3] == __builtin_bit_cast(unsigned, d);
}

constexpr bool match_m128d(__m128d _v, double a, double b) {
  return ((__v2du)_v)[0] == __builtin_bit_cast(unsigned long long, a) && ((__v2du)_v)[1] == __builtin_bit_cast(unsigned long long, b);
}

#ifdef __SSE2__
constexpr bool match_m128h(__m128h _v, _Float16 __e00, _Float16 __e01, _Float16 __e02, _Float16 __e03, _Float16 __e04, _Float16 __e05, _Float16 __e06, _Float16 __e07) {
  return ((__v8hu)_v)[ 0] == __builtin_bit_cast(unsigned short, __e00) && ((__v8hu)_v)[ 1] == __builtin_bit_cast(unsigned short, __e01) && ((__v8hu)_v)[ 2] == __builtin_bit_cast(unsigned short, __e02) && ((__v8hu)_v)[ 3] == __builtin_bit_cast(unsigned short, __e03) &&
         ((__v8hu)_v)[ 4] == __builtin_bit_cast(unsigned short, __e04) && ((__v8hu)_v)[ 5] == __builtin_bit_cast(unsigned short, __e05) && ((__v8hu)_v)[ 6] == __builtin_bit_cast(unsigned short, __e06) && ((__v8hu)_v)[ 7] == __builtin_bit_cast(unsigned short, __e07);
}
#endif

constexpr bool match_m128i(__m128i _v, unsigned long long a, unsigned long long b) {
  return ((__v2du)_v)[0] == a && ((__v2du)_v)[1] == b;
}

constexpr bool match_v2di(__m128i v, long long a, long long b) {
  return v[0] == a && v[1] == b;
}

constexpr bool match_v2du(__m128i _v, unsigned long long a, unsigned long long b) {
  return ((__v2du)_v)[0] == a && ((__v2du)_v)[1] == b;
}

constexpr bool match_v4si(__m128i _v, int a, int b, int c, int d) {
  return ((__v4si)_v)[0] == a && ((__v4si)_v)[1] == b && ((__v4si)_v)[2] == c && ((__v4si)_v)[3] == d;
}

constexpr bool match_v4su(__m128i _v, unsigned a, unsigned b, unsigned c, unsigned d) {
  return ((__v4su)_v)[0] == a && ((__v4su)_v)[1] == b && ((__v4su)_v)[2] == c && ((__v4su)_v)[3] == d;
}

constexpr bool match_v8hi(__m128i _v, short a, short b, short c, short d, short e, short f, short g, short h) {
  return ((__v8hi)_v)[0] == a && ((__v8hi)_v)[1] == b && ((__v8hi)_v)[2] == c && ((__v8hi)_v)[3] == d && ((__v8hi)_v)[4] == e && ((__v8hi)_v)[5] == f && ((__v8hi)_v)[6] == g && ((__v8hi)_v)[7] == h;
}

constexpr bool match_v8hu(__m128i _v, unsigned short a, unsigned short b, unsigned short c, unsigned short d, unsigned short e, unsigned short f, unsigned short g, unsigned short h) {
  return ((__v8hu)_v)[0] == a && ((__v8hu)_v)[1] == b && ((__v8hu)_v)[2] == c && ((__v8hu)_v)[3] == d && ((__v8hu)_v)[4] == e && ((__v8hu)_v)[5] == f && ((__v8hu)_v)[6] == g && ((__v8hu)_v)[7] == h;
}

constexpr bool match_v16qi(__m128i _v, signed char a, signed char b, signed char c, signed char d, signed char e, signed char f, signed char g, signed char h, signed char i, signed char j, signed char k, signed char l, signed char m, signed char n, signed char o, signed char p) {
  return ((__v16qs)_v)[0] == a && ((__v16qs)_v)[1] == b && ((__v16qs)_v)[2] == c && ((__v16qs)_v)[3] == d && ((__v16qs)_v)[4] == e && ((__v16qs)_v)[5] == f && ((__v16qs)_v)[6] == g && ((__v16qs)_v)[7] == h && ((__v16qs)_v)[8] == i && ((__v16qs)_v)[9] == j && ((__v16qs)_v)[10] == k && ((__v16qs)_v)[11] == l && ((__v16qs)_v)[12] == m && ((__v16qs)_v)[13] == n && ((__v16qs)_v)[14] == o && ((__v16qs)_v)[15] == p;
}

constexpr bool match_v16qu(__m128i _v, unsigned char a, unsigned char b, unsigned char c, unsigned char d, unsigned char e, unsigned char f, unsigned char g, unsigned char h, unsigned char i, unsigned char j, unsigned char k, unsigned char l, unsigned char m, unsigned char n, unsigned char o, unsigned char p) {
  return ((__v16qu)_v)[0] == a && ((__v16qu)_v)[1] == b && ((__v16qu)_v)[2] == c && ((__v16qu)_v)[3] == d && ((__v16qu)_v)[4] == e && ((__v16qu)_v)[5] == f && ((__v16qu)_v)[6] == g && ((__v16qu)_v)[7] == h && ((__v16qu)_v)[8] == i && ((__v16qu)_v)[9] == j && ((__v16qu)_v)[10] == k && ((__v16qu)_v)[11] == l && ((__v16qu)_v)[12] == m && ((__v16qu)_v)[13] == n && ((__v16qu)_v)[14] == o && ((__v16qu)_v)[15] == p;
}

constexpr bool match_m256(__m256 _v, float __e00, float __e01, float __e02, float __e03, float __e04, float __e05, float __e06, float __e07) {
  return ((__v8su)_v)[ 0] == __builtin_bit_cast(unsigned, __e00) && ((__v8su)_v)[ 1] == __builtin_bit_cast(unsigned, __e01) && ((__v8su)_v)[ 2] == __builtin_bit_cast(unsigned, __e02) && ((__v8su)_v)[ 3] == __builtin_bit_cast(unsigned, __e03) &&
         ((__v8su)_v)[ 4] == __builtin_bit_cast(unsigned, __e04) && ((__v8su)_v)[ 5] == __builtin_bit_cast(unsigned, __e05) && ((__v8su)_v)[ 6] == __builtin_bit_cast(unsigned, __e06) && ((__v8su)_v)[ 7] == __builtin_bit_cast(unsigned, __e07);
}

constexpr bool match_m256d(__m256d _v, double a, double b, double c, double d) {
  return ((__v4du)_v)[0] == __builtin_bit_cast(unsigned long long, a) && ((__v4du)_v)[1] == __builtin_bit_cast(unsigned long long, b) && ((__v4du)_v)[2] == __builtin_bit_cast(unsigned long long, c) && ((__v4du)_v)[3] == __builtin_bit_cast(unsigned long long, d);
}

#ifdef __SSE2__
constexpr bool match_m256h(__m256h _v, _Float16 __e00, _Float16 __e01, _Float16 __e02, _Float16 __e03, _Float16 __e04, _Float16 __e05, _Float16 __e06, _Float16 __e07,
                                       _Float16 __e08, _Float16 __e09, _Float16 __e10, _Float16 __e11, _Float16 __e12, _Float16 __e13, _Float16 __e14, _Float16 __e15) {
  return ((__v16hu)_v)[ 0] == __builtin_bit_cast(unsigned short, __e00) && ((__v16hu)_v)[ 1] == __builtin_bit_cast(unsigned short, __e01) && ((__v16hu)_v)[ 2] == __builtin_bit_cast(unsigned short, __e02) && ((__v16hu)_v)[ 3] == __builtin_bit_cast(unsigned short, __e03) &&
         ((__v16hu)_v)[ 4] == __builtin_bit_cast(unsigned short, __e04) && ((__v16hu)_v)[ 5] == __builtin_bit_cast(unsigned short, __e05) && ((__v16hu)_v)[ 6] == __builtin_bit_cast(unsigned short, __e06) && ((__v16hu)_v)[ 7] == __builtin_bit_cast(unsigned short, __e07) &&
         ((__v16hu)_v)[ 8] == __builtin_bit_cast(unsigned short, __e08) && ((__v16hu)_v)[ 9] == __builtin_bit_cast(unsigned short, __e09) && ((__v16hu)_v)[10] == __builtin_bit_cast(unsigned short, __e10) && ((__v16hu)_v)[11] == __builtin_bit_cast(unsigned short, __e11) &&
         ((__v16hu)_v)[12] == __builtin_bit_cast(unsigned short, __e12) && ((__v16hu)_v)[13] == __builtin_bit_cast(unsigned short, __e13) && ((__v16hu)_v)[14] == __builtin_bit_cast(unsigned short, __e14) && ((__v16hu)_v)[15] == __builtin_bit_cast(unsigned short, __e15);
}
#endif

constexpr bool match_m256i(__m256i _v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d) {
  return ((__v4du)_v)[0] == a && ((__v4du)_v)[1] == b && ((__v4du)_v)[2] == c && ((__v4du)_v)[3] == d;
}

constexpr bool match_v4di(__m256i _v, long long a, long long b, long long c, long long d) {
  return ((__v4di)_v)[0] == a && ((__v4di)_v)[1] == b && ((__v4di)_v)[2] == c && ((__v4di)_v)[3] == d;
}

constexpr bool match_v8si(__m256i _v, int a, int b, int c, int d, int e, int f, int g, int h) {
  return ((__v8si)_v)[0] == a && ((__v8si)_v)[1] == b && ((__v8si)_v)[2] == c && ((__v8si)_v)[3] == d && ((__v8si)_v)[4] == e && ((__v8si)_v)[5] == f && ((__v8si)_v)[6] == g && ((__v8si)_v)[7] == h;
}

constexpr bool match_v8su(__m256i _v, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e, unsigned f, unsigned g, unsigned h) {
  return ((__v8su)_v)[0] == a && ((__v8su)_v)[1] == b && ((__v8su)_v)[2] == c && ((__v8su)_v)[3] == d && ((__v8su)_v)[4] == e && ((__v8su)_v)[5] == f && ((__v8su)_v)[6] == g && ((__v8su)_v)[7] == h;
}

constexpr bool match_v16hi(__m256i _v, short a, short b, short c, short d, short e, short f, short g, short h, short i, short j, short k, short l, short m, short n, short o, short p) {
  return ((__v16hi)_v)[0] == a && ((__v16hi)_v)[1] == b && ((__v16hi)_v)[2] == c && ((__v16hi)_v)[3] == d && ((__v16hi)_v)[4] == e && ((__v16hi)_v)[5] == f && ((__v16hi)_v)[6] == g && ((__v16hi)_v)[7] == h && ((__v16hi)_v)[8] == i && ((__v16hi)_v)[9] == j && ((__v16hi)_v)[10] == k && ((__v16hi)_v)[11] == l && ((__v16hi)_v)[12] == m && ((__v16hi)_v)[13] == n && ((__v16hi)_v)[14] == o && ((__v16hi)_v)[15] == p;
}

constexpr bool match_v16hu(__m256i _v, unsigned short a, unsigned short b, unsigned short c, unsigned short d, unsigned short e, unsigned short f, unsigned short g, unsigned short h, unsigned short i, unsigned short j, unsigned short k, unsigned short l, unsigned short m, unsigned short n, unsigned short o, unsigned short p) {
  return ((__v16hu)_v)[0] == a && ((__v16hu)_v)[1] == b && ((__v16hu)_v)[2] == c && ((__v16hu)_v)[3] == d && ((__v16hu)_v)[4] == e && ((__v16hu)_v)[5] == f && ((__v16hu)_v)[6] == g && ((__v16hu)_v)[7] == h && ((__v16hu)_v)[8] == i && ((__v16hu)_v)[9] == j && ((__v16hu)_v)[10] == k && ((__v16hu)_v)[11] == l && ((__v16hu)_v)[12] == m && ((__v16hu)_v)[13] == n && ((__v16hu)_v)[14] == o && ((__v16hu)_v)[15] == p;
}

constexpr bool match_v32qi(__m256i _v, signed char __b00, signed char __b01, signed char __b02, signed char __b03, signed char __b04, signed char __b05, signed char __b06, signed char __b07,
                                       signed char __b08, signed char __b09, signed char __b10, signed char __b11, signed char __b12, signed char __b13, signed char __b14, signed char __b15,
                                       signed char __b16, signed char __b17, signed char __b18, signed char __b19, signed char __b20, signed char __b21, signed char __b22, signed char __b23,
                                       signed char __b24, signed char __b25, signed char __b26, signed char __b27, signed char __b28, signed char __b29, signed char __b30, signed char __b31) {
  return ((__v32qs)_v)[ 0] == __b00 && ((__v32qs)_v)[ 1] == __b01 && ((__v32qs)_v)[ 2] == __b02 && ((__v32qs)_v)[ 3] == __b03 && ((__v32qs)_v)[ 4] == __b04 && ((__v32qs)_v)[ 5] == __b05 && ((__v32qs)_v)[ 6] == __b06 && ((__v32qs)_v)[ 7] ==  __b07 &&
         ((__v32qs)_v)[ 8] == __b08 && ((__v32qs)_v)[ 9] == __b09 && ((__v32qs)_v)[10] == __b10 && ((__v32qs)_v)[11] == __b11 && ((__v32qs)_v)[12] == __b12 && ((__v32qs)_v)[13] == __b13 && ((__v32qs)_v)[14] == __b14 && ((__v32qs)_v)[15] ==  __b15 &&
         ((__v32qs)_v)[16] == __b16 && ((__v32qs)_v)[17] == __b17 && ((__v32qs)_v)[18] == __b18 && ((__v32qs)_v)[19] == __b19 && ((__v32qs)_v)[20] == __b20 && ((__v32qs)_v)[21] == __b21 && ((__v32qs)_v)[22] == __b22 && ((__v32qs)_v)[23] ==  __b23 &&
         ((__v32qs)_v)[24] == __b24 && ((__v32qs)_v)[25] == __b25 && ((__v32qs)_v)[26] == __b26 && ((__v32qs)_v)[27] == __b27 && ((__v32qs)_v)[28] == __b28 && ((__v32qs)_v)[29] == __b29 && ((__v32qs)_v)[30] == __b30 && ((__v32qs)_v)[31] ==  __b31;
}

constexpr bool match_v32qu(__m256i _v, unsigned char __b00, unsigned char __b01, unsigned char __b02, unsigned char __b03, unsigned char __b04, unsigned char __b05, unsigned char __b06, unsigned char __b07,
                                       unsigned char __b08, unsigned char __b09, unsigned char __b10, unsigned char __b11, unsigned char __b12, unsigned char __b13, unsigned char __b14, unsigned char __b15,
                                       unsigned char __b16, unsigned char __b17, unsigned char __b18, unsigned char __b19, unsigned char __b20, unsigned char __b21, unsigned char __b22, unsigned char __b23,
                                       unsigned char __b24, unsigned char __b25, unsigned char __b26, unsigned char __b27, unsigned char __b28, unsigned char __b29, unsigned char __b30, unsigned char __b31) {
  return ((__v32qu)_v)[ 0] == __b00 && ((__v32qu)_v)[ 1] == __b01 && ((__v32qu)_v)[ 2] == __b02 && ((__v32qu)_v)[ 3] == __b03 && ((__v32qu)_v)[ 4] == __b04 && ((__v32qu)_v)[ 5] == __b05 && ((__v32qu)_v)[ 6] == __b06 && ((__v32qu)_v)[ 7] ==  __b07 &&
         ((__v32qu)_v)[ 8] == __b08 && ((__v32qu)_v)[ 9] == __b09 && ((__v32qu)_v)[10] == __b10 && ((__v32qu)_v)[11] == __b11 && ((__v32qu)_v)[12] == __b12 && ((__v32qu)_v)[13] == __b13 && ((__v32qu)_v)[14] == __b14 && ((__v32qu)_v)[15] ==  __b15 &&
         ((__v32qu)_v)[16] == __b16 && ((__v32qu)_v)[17] == __b17 && ((__v32qu)_v)[18] == __b18 && ((__v32qu)_v)[19] == __b19 && ((__v32qu)_v)[20] == __b20 && ((__v32qu)_v)[21] == __b21 && ((__v32qu)_v)[22] == __b22 && ((__v32qu)_v)[23] ==  __b23 &&
         ((__v32qu)_v)[24] == __b24 && ((__v32qu)_v)[25] == __b25 && ((__v32qu)_v)[26] == __b26 && ((__v32qu)_v)[27] == __b27 && ((__v32qu)_v)[28] == __b28 && ((__v32qu)_v)[29] == __b29 && ((__v32qu)_v)[30] == __b30 && ((__v32qu)_v)[31] ==  __b31;
}

constexpr bool match_m512(__m512 _v, float __e00, float __e01, float __e02, float __e03, float __e04, float __e05, float __e06, float __e07, float __e08, float __e09, float __e10, float __e11, float __e12, float __e13, float __e14, float __e15) {
  return ((__v16su)_v)[ 0] == __builtin_bit_cast(unsigned, __e00) && ((__v16su)_v)[ 1] == __builtin_bit_cast(unsigned, __e01) && ((__v16su)_v)[ 2] == __builtin_bit_cast(unsigned, __e02) && ((__v16su)_v)[ 3] == __builtin_bit_cast(unsigned, __e03) &&
         ((__v16su)_v)[ 4] == __builtin_bit_cast(unsigned, __e04) && ((__v16su)_v)[ 5] == __builtin_bit_cast(unsigned, __e05) && ((__v16su)_v)[ 6] == __builtin_bit_cast(unsigned, __e06) && ((__v16su)_v)[ 7] == __builtin_bit_cast(unsigned, __e07) &&
         ((__v16su)_v)[ 8] == __builtin_bit_cast(unsigned, __e08) && ((__v16su)_v)[ 9] == __builtin_bit_cast(unsigned, __e09) && ((__v16su)_v)[10] == __builtin_bit_cast(unsigned, __e10) && ((__v16su)_v)[11] == __builtin_bit_cast(unsigned, __e11) &&
         ((__v16su)_v)[12] == __builtin_bit_cast(unsigned, __e12) && ((__v16su)_v)[13] == __builtin_bit_cast(unsigned, __e13) && ((__v16su)_v)[14] == __builtin_bit_cast(unsigned, __e14) && ((__v16su)_v)[15] == __builtin_bit_cast(unsigned, __e15);
}

constexpr bool match_m512d(__m512d _v, double __e00, double __e01, double __e02, double __e03, double __e04, double __e05, double __e06, double __e07) {
  return ((__v8du)_v)[ 0] == __builtin_bit_cast(unsigned long long, __e00) && ((__v8du)_v)[ 1] == __builtin_bit_cast(unsigned long long, __e01) && ((__v8du)_v)[ 2] == __builtin_bit_cast(unsigned long long, __e02) && ((__v8du)_v)[ 3] == __builtin_bit_cast(unsigned long long, __e03) &&
         ((__v8du)_v)[ 4] == __builtin_bit_cast(unsigned long long, __e04) && ((__v8du)_v)[ 5] == __builtin_bit_cast(unsigned long long, __e05) && ((__v8du)_v)[ 6] == __builtin_bit_cast(unsigned long long, __e06) && ((__v8du)_v)[ 7] == __builtin_bit_cast(unsigned long long, __e07);
}

#ifdef __SSE2__
constexpr bool match_m512h(__m512h _v, _Float16 __e00, _Float16 __e01, _Float16 __e02, _Float16 __e03, _Float16 __e04, _Float16 __e05, _Float16 __e06, _Float16 __e07,
                                       _Float16 __e08, _Float16 __e09, _Float16 __e10, _Float16 __e11, _Float16 __e12, _Float16 __e13, _Float16 __e14, _Float16 __e15,
                                       _Float16 __e16, _Float16 __e17, _Float16 __e18, _Float16 __e19, _Float16 __e20, _Float16 __e21, _Float16 __e22, _Float16 __e23,
                                       _Float16 __e24, _Float16 __e25, _Float16 __e26, _Float16 __e27, _Float16 __e28, _Float16 __e29, _Float16 __e30, _Float16 __e31) {
  return ((__v32hu)_v)[ 0] == __builtin_bit_cast(unsigned short, __e00) && ((__v32hu)_v)[ 1] == __builtin_bit_cast(unsigned short, __e01) && ((__v32hu)_v)[ 2] == __builtin_bit_cast(unsigned short, __e02) && ((__v32hu)_v)[ 3] == __builtin_bit_cast(unsigned short, __e03) &&
         ((__v32hu)_v)[ 4] == __builtin_bit_cast(unsigned short, __e04) && ((__v32hu)_v)[ 5] == __builtin_bit_cast(unsigned short, __e05) && ((__v32hu)_v)[ 6] == __builtin_bit_cast(unsigned short, __e06) && ((__v32hu)_v)[ 7] == __builtin_bit_cast(unsigned short, __e07) &&
         ((__v32hu)_v)[ 8] == __builtin_bit_cast(unsigned short, __e08) && ((__v32hu)_v)[ 9] == __builtin_bit_cast(unsigned short, __e09) && ((__v32hu)_v)[10] == __builtin_bit_cast(unsigned short, __e10) && ((__v32hu)_v)[11] == __builtin_bit_cast(unsigned short, __e11) &&
         ((__v32hu)_v)[12] == __builtin_bit_cast(unsigned short, __e12) && ((__v32hu)_v)[13] == __builtin_bit_cast(unsigned short, __e13) && ((__v32hu)_v)[14] == __builtin_bit_cast(unsigned short, __e14) && ((__v32hu)_v)[15] == __builtin_bit_cast(unsigned short, __e15) &&
         ((__v32hu)_v)[16] == __builtin_bit_cast(unsigned short, __e16) && ((__v32hu)_v)[17] == __builtin_bit_cast(unsigned short, __e17) && ((__v32hu)_v)[18] == __builtin_bit_cast(unsigned short, __e18) && ((__v32hu)_v)[19] == __builtin_bit_cast(unsigned short, __e19) &&
         ((__v32hu)_v)[20] == __builtin_bit_cast(unsigned short, __e20) && ((__v32hu)_v)[21] == __builtin_bit_cast(unsigned short, __e21) && ((__v32hu)_v)[22] == __builtin_bit_cast(unsigned short, __e22) && ((__v32hu)_v)[23] == __builtin_bit_cast(unsigned short, __e23) &&
         ((__v32hu)_v)[24] == __builtin_bit_cast(unsigned short, __e24) && ((__v32hu)_v)[25] == __builtin_bit_cast(unsigned short, __e25) && ((__v32hu)_v)[26] == __builtin_bit_cast(unsigned short, __e26) && ((__v32hu)_v)[27] == __builtin_bit_cast(unsigned short, __e27) &&
         ((__v32hu)_v)[28] == __builtin_bit_cast(unsigned short, __e28) && ((__v32hu)_v)[29] == __builtin_bit_cast(unsigned short, __e29) && ((__v32hu)_v)[30] == __builtin_bit_cast(unsigned short, __e30) && ((__v32hu)_v)[31] == __builtin_bit_cast(unsigned short, __e31);
}
#endif

constexpr bool match_m512i(__m512i _v, unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d, unsigned long long e, unsigned long long f, unsigned long long g, unsigned long long h) {
  return ((__v8du)_v)[0] == a && ((__v8du)_v)[1] == b && ((__v8du)_v)[2] == c && ((__v8du)_v)[3] == d && ((__v8du)_v)[4] == e && ((__v8du)_v)[5] == f && ((__v8du)_v)[6] == g && ((__v8du)_v)[7] == h;
}

constexpr bool match_v8di(__m512i _v, long long a, long long b, long long c, long long d, long long e, long long f, long long g, long long h) {
  return ((__v8di)_v)[0] == a && ((__v8di)_v)[1] == b && ((__v8di)_v)[2] == c && ((__v8di)_v)[3] == d && ((__v8di)_v)[4] == e && ((__v8di)_v)[5] == f && ((__v8di)_v)[6] == g && ((__v8di)_v)[7] == h;
}

constexpr bool match_v16si(__m512i _v, int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o, int p) {
  return ((__v16si)_v)[0] == a && ((__v16si)_v)[1] == b && ((__v16si)_v)[2] == c && ((__v16si)_v)[3] == d && ((__v16si)_v)[4] == e && ((__v16si)_v)[5] == f && ((__v16si)_v)[6] == g && ((__v16si)_v)[7] == h && ((__v16si)_v)[8] == i && ((__v16si)_v)[9] == j && ((__v16si)_v)[10] == k && ((__v16si)_v)[11] == l && ((__v16si)_v)[12] == m && ((__v16si)_v)[13] == n && ((__v16si)_v)[14] == o && ((__v16si)_v)[15] == p;
}

constexpr bool match_v32hi(__m512i _v, short __e00, short __e01, short __e02, short __e03, short __e04, short __e05, short __e06, short __e07,
                                       short __e08, short __e09, short __e10, short __e11, short __e12, short __e13, short __e14, short __e15,
                                       short __e16, short __e17, short __e18, short __e19, short __e20, short __e21, short __e22, short __e23,
                                       short __e24, short __e25, short __e26, short __e27, short __e28, short __e29, short __e30, short __e31) {
  return ((__v32hi)_v)[ 0] == __e00 && ((__v32hi)_v)[ 1] == __e01 && ((__v32hi)_v)[ 2] == __e02 && ((__v32hi)_v)[ 3] == __e03 && ((__v32hi)_v)[ 4] == __e04 && ((__v32hi)_v)[ 5] == __e05 && ((__v32hi)_v)[ 6] == __e06 && ((__v32hi)_v)[ 7] ==  __e07 &&
         ((__v32hi)_v)[ 8] == __e08 && ((__v32hi)_v)[ 9] == __e09 && ((__v32hi)_v)[10] == __e10 && ((__v32hi)_v)[11] == __e11 && ((__v32hi)_v)[12] == __e12 && ((__v32hi)_v)[13] == __e13 && ((__v32hi)_v)[14] == __e14 && ((__v32hi)_v)[15] ==  __e15 &&
         ((__v32hi)_v)[16] == __e16 && ((__v32hi)_v)[17] == __e17 && ((__v32hi)_v)[18] == __e18 && ((__v32hi)_v)[19] == __e19 && ((__v32hi)_v)[20] == __e20 && ((__v32hi)_v)[21] == __e21 && ((__v32hi)_v)[22] == __e22 && ((__v32hi)_v)[23] ==  __e23 &&
         ((__v32hi)_v)[24] == __e24 && ((__v32hi)_v)[25] == __e25 && ((__v32hi)_v)[26] == __e26 && ((__v32hi)_v)[27] == __e27 && ((__v32hi)_v)[28] == __e28 && ((__v32hi)_v)[29] == __e29 && ((__v32hi)_v)[30] == __e30 && ((__v32hi)_v)[31] ==  __e31;
}

constexpr bool match_v32hu(__m512i _v, unsigned short __e00, unsigned short __e01, unsigned short __e02, unsigned short __e03, unsigned short __e04, unsigned short __e05, unsigned short __e06, unsigned short __e07,
                                       unsigned short __e08, unsigned short __e09, unsigned short __e10, unsigned short __e11, unsigned short __e12, unsigned short __e13, unsigned short __e14, unsigned short __e15,
                                       unsigned short __e16, unsigned short __e17, unsigned short __e18, unsigned short __e19, unsigned short __e20, unsigned short __e21, unsigned short __e22, unsigned short __e23,
                                       unsigned short __e24, unsigned short __e25, unsigned short __e26, unsigned short __e27, unsigned short __e28, unsigned short __e29, unsigned short __e30, unsigned short __e31) {
  return ((__v32hu)_v)[ 0] == __e00 && ((__v32hu)_v)[ 1] == __e01 && ((__v32hu)_v)[ 2] == __e02 && ((__v32hu)_v)[ 3] == __e03 && ((__v32hu)_v)[ 4] == __e04 && ((__v32hu)_v)[ 5] == __e05 && ((__v32hu)_v)[ 6] == __e06 && ((__v32hu)_v)[ 7] ==  __e07 &&
         ((__v32hu)_v)[ 8] == __e08 && ((__v32hu)_v)[ 9] == __e09 && ((__v32hu)_v)[10] == __e10 && ((__v32hu)_v)[11] == __e11 && ((__v32hu)_v)[12] == __e12 && ((__v32hu)_v)[13] == __e13 && ((__v32hu)_v)[14] == __e14 && ((__v32hu)_v)[15] ==  __e15 &&
         ((__v32hu)_v)[16] == __e16 && ((__v32hu)_v)[17] == __e17 && ((__v32hu)_v)[18] == __e18 && ((__v32hu)_v)[19] == __e19 && ((__v32hu)_v)[20] == __e20 && ((__v32hu)_v)[21] == __e21 && ((__v32hu)_v)[22] == __e22 && ((__v32hu)_v)[23] ==  __e23 &&
         ((__v32hu)_v)[24] == __e24 && ((__v32hu)_v)[25] == __e25 && ((__v32hu)_v)[26] == __e26 && ((__v32hu)_v)[27] == __e27 && ((__v32hu)_v)[28] == __e28 && ((__v32hu)_v)[29] == __e29 && ((__v32hu)_v)[30] == __e30 && ((__v32hu)_v)[31] ==  __e31;
}

constexpr bool match_v64qi(__m512i _v, signed char __e00, signed char __e01, signed char __e02, signed char __e03, signed char __e04, signed char __e05, signed char __e06, signed char __e07,
                                       signed char __e08, signed char __e09, signed char __e10, signed char __e11, signed char __e12, signed char __e13, signed char __e14, signed char __e15,
                                       signed char __e16, signed char __e17, signed char __e18, signed char __e19, signed char __e20, signed char __e21, signed char __e22, signed char __e23,
                                       signed char __e24, signed char __e25, signed char __e26, signed char __e27, signed char __e28, signed char __e29, signed char __e30, signed char __e31,
                                       signed char __e32, signed char __e33, signed char __e34, signed char __e35, signed char __e36, signed char __e37, signed char __e38, signed char __e39,
                                       signed char __e40, signed char __e41, signed char __e42, signed char __e43, signed char __e44, signed char __e45, signed char __e46, signed char __e47,
                                       signed char __e48, signed char __e49, signed char __e50, signed char __e51, signed char __e52, signed char __e53, signed char __e54, signed char __e55,
                                       signed char __e56, signed char __e57, signed char __e58, signed char __e59, signed char __e60, signed char __e61, signed char __e62, signed char __e63) {
  return ((__v64qs)_v)[ 0] == __e00 && ((__v64qs)_v)[ 1] == __e01 && ((__v64qs)_v)[ 2] == __e02 && ((__v64qs)_v)[ 3] == __e03 && ((__v64qs)_v)[ 4] == __e04 && ((__v64qs)_v)[ 5] == __e05 && ((__v64qs)_v)[ 6] == __e06 && ((__v64qs)_v)[ 7] == __e07 &&
         ((__v64qs)_v)[ 8] == __e08 && ((__v64qs)_v)[ 9] == __e09 && ((__v64qs)_v)[10] == __e10 && ((__v64qs)_v)[11] == __e11 && ((__v64qs)_v)[12] == __e12 && ((__v64qs)_v)[13] == __e13 && ((__v64qs)_v)[14] == __e14 && ((__v64qs)_v)[15] == __e15 &&
         ((__v64qs)_v)[16] == __e16 && ((__v64qs)_v)[17] == __e17 && ((__v64qs)_v)[18] == __e18 && ((__v64qs)_v)[19] == __e19 && ((__v64qs)_v)[20] == __e20 && ((__v64qs)_v)[21] == __e21 && ((__v64qs)_v)[22] == __e22 && ((__v64qs)_v)[23] == __e23 &&
         ((__v64qs)_v)[24] == __e24 && ((__v64qs)_v)[25] == __e25 && ((__v64qs)_v)[26] == __e26 && ((__v64qs)_v)[27] == __e27 && ((__v64qs)_v)[28] == __e28 && ((__v64qs)_v)[29] == __e29 && ((__v64qs)_v)[30] == __e30 && ((__v64qs)_v)[31] == __e31 &&
         ((__v64qs)_v)[32] == __e32 && ((__v64qs)_v)[33] == __e33 && ((__v64qs)_v)[34] == __e34 && ((__v64qs)_v)[35] == __e35 && ((__v64qs)_v)[36] == __e36 && ((__v64qs)_v)[37] == __e37 && ((__v64qs)_v)[38] == __e38 && ((__v64qs)_v)[39] == __e39 &&
         ((__v64qs)_v)[40] == __e40 && ((__v64qs)_v)[41] == __e41 && ((__v64qs)_v)[42] == __e42 && ((__v64qs)_v)[43] == __e43 && ((__v64qs)_v)[44] == __e44 && ((__v64qs)_v)[45] == __e45 && ((__v64qs)_v)[46] == __e46 && ((__v64qs)_v)[47] == __e47 &&
         ((__v64qs)_v)[48] == __e48 && ((__v64qs)_v)[49] == __e49 && ((__v64qs)_v)[50] == __e50 && ((__v64qs)_v)[51] == __e51 && ((__v64qs)_v)[52] == __e52 && ((__v64qs)_v)[53] == __e53 && ((__v64qs)_v)[54] == __e54 && ((__v64qs)_v)[55] == __e55 &&
         ((__v64qs)_v)[56] == __e56 && ((__v64qs)_v)[57] == __e57 && ((__v64qs)_v)[58] == __e58 && ((__v64qs)_v)[59] == __e59 && ((__v64qs)_v)[60] == __e60 && ((__v64qs)_v)[61] == __e61 && ((__v64qs)_v)[62] == __e62 && ((__v64qs)_v)[63] == __e63;
}

constexpr bool match_v64qu(__m512i _v, unsigned char __e00, unsigned char __e01, unsigned char __e02, unsigned char __e03, unsigned char __e04, unsigned char __e05, unsigned char __e06, unsigned char __e07,
                                       unsigned char __e08, unsigned char __e09, unsigned char __e10, unsigned char __e11, unsigned char __e12, unsigned char __e13, unsigned char __e14, unsigned char __e15,
                                       unsigned char __e16, unsigned char __e17, unsigned char __e18, unsigned char __e19, unsigned char __e20, unsigned char __e21, unsigned char __e22, unsigned char __e23,
                                       unsigned char __e24, unsigned char __e25, unsigned char __e26, unsigned char __e27, unsigned char __e28, unsigned char __e29, unsigned char __e30, unsigned char __e31,
                                       unsigned char __e32, unsigned char __e33, unsigned char __e34, unsigned char __e35, unsigned char __e36, unsigned char __e37, unsigned char __e38, unsigned char __e39,
                                       unsigned char __e40, unsigned char __e41, unsigned char __e42, unsigned char __e43, unsigned char __e44, unsigned char __e45, unsigned char __e46, unsigned char __e47,
                                       unsigned char __e48, unsigned char __e49, unsigned char __e50, unsigned char __e51, unsigned char __e52, unsigned char __e53, unsigned char __e54, unsigned char __e55,
                                       unsigned char __e56, unsigned char __e57, unsigned char __e58, unsigned char __e59, unsigned char __e60, unsigned char __e61, unsigned char __e62, unsigned char __e63) {
  return ((__v64qu)_v)[ 0] == __e00 && ((__v64qu)_v)[ 1] == __e01 && ((__v64qu)_v)[ 2] == __e02 && ((__v64qu)_v)[ 3] == __e03 && ((__v64qu)_v)[ 4] == __e04 && ((__v64qu)_v)[ 5] == __e05 && ((__v64qu)_v)[ 6] == __e06 && ((__v64qu)_v)[ 7] == __e07 &&
         ((__v64qu)_v)[ 8] == __e08 && ((__v64qu)_v)[ 9] == __e09 && ((__v64qu)_v)[10] == __e10 && ((__v64qu)_v)[11] == __e11 && ((__v64qu)_v)[12] == __e12 && ((__v64qu)_v)[13] == __e13 && ((__v64qu)_v)[14] == __e14 && ((__v64qu)_v)[15] == __e15 &&
         ((__v64qu)_v)[16] == __e16 && ((__v64qu)_v)[17] == __e17 && ((__v64qu)_v)[18] == __e18 && ((__v64qu)_v)[19] == __e19 && ((__v64qu)_v)[20] == __e20 && ((__v64qu)_v)[21] == __e21 && ((__v64qu)_v)[22] == __e22 && ((__v64qu)_v)[23] == __e23 &&
         ((__v64qu)_v)[24] == __e24 && ((__v64qu)_v)[25] == __e25 && ((__v64qu)_v)[26] == __e26 && ((__v64qu)_v)[27] == __e27 && ((__v64qu)_v)[28] == __e28 && ((__v64qu)_v)[29] == __e29 && ((__v64qu)_v)[30] == __e30 && ((__v64qu)_v)[31] == __e31 &&
         ((__v64qu)_v)[32] == __e32 && ((__v64qu)_v)[33] == __e33 && ((__v64qu)_v)[34] == __e34 && ((__v64qu)_v)[35] == __e35 && ((__v64qu)_v)[36] == __e36 && ((__v64qu)_v)[37] == __e37 && ((__v64qu)_v)[38] == __e38 && ((__v64qu)_v)[39] == __e39 &&
         ((__v64qu)_v)[40] == __e40 && ((__v64qu)_v)[41] == __e41 && ((__v64qu)_v)[42] == __e42 && ((__v64qu)_v)[43] == __e43 && ((__v64qu)_v)[44] == __e44 && ((__v64qu)_v)[45] == __e45 && ((__v64qu)_v)[46] == __e46 && ((__v64qu)_v)[47] == __e47 &&
         ((__v64qu)_v)[48] == __e48 && ((__v64qu)_v)[49] == __e49 && ((__v64qu)_v)[50] == __e50 && ((__v64qu)_v)[51] == __e51 && ((__v64qu)_v)[52] == __e52 && ((__v64qu)_v)[53] == __e53 && ((__v64qu)_v)[54] == __e54 && ((__v64qu)_v)[55] == __e55 &&
         ((__v64qu)_v)[56] == __e56 && ((__v64qu)_v)[57] == __e57 && ((__v64qu)_v)[58] == __e58 && ((__v64qu)_v)[59] == __e59 && ((__v64qu)_v)[60] == __e60 && ((__v64qu)_v)[61] == __e61 && ((__v64qu)_v)[62] == __e62 && ((__v64qu)_v)[63] == __e63;
}


#else

#define TEST_CONSTEXPR(...)

#endif
