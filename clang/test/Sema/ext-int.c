// RUN: %clang_cc1 -fsyntax-only -verify %s -Wimplicit-int-conversion -Wno-unused -triple x86_64-gnu-linux

typedef _BitInt(31) EI31;

void Ternary(_BitInt(30) s30, EI31 s31a, _BitInt(31) s31b,
             _BitInt(32) s32, int b) {
  b ? s30 : s31a;
  b ? s31a : s30;
  b ? s32 : 0;
  (void)(b ? s31a : s31b);
  (void)(s30 ? s31a : s31b);
}

struct CursedBitField {
  _BitInt(4) A : 8; // expected-error {{width of bit-field 'A' (8 bits) exceeds the width of its type (4 bits)}}
};

#define EXPR_HAS_TYPE(EXPR, TYPE) _Generic((EXPR), default : 0, TYPE : 1)

void Ops(void) {
  _BitInt(4) x4_s = 1;
  _BitInt(32) x32_s = 1;
  _BitInt(43) x43_s = 1;
  unsigned _BitInt(4) x4_u = 1;
  unsigned _BitInt(43) x43_u = 1;
  unsigned _BitInt(32) x32_u = 1;
  int x_int = 1;
  unsigned x_uint = 1;

  // Same size/sign ops don't change type.
  _Static_assert(EXPR_HAS_TYPE(x43_s + x43_s, _BitInt(43)), "");
  _Static_assert(EXPR_HAS_TYPE(x4_s - x4_s, _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(x43_u * x43_u, unsigned _BitInt(43)), "");
  _Static_assert(EXPR_HAS_TYPE(x4_u / x4_u, unsigned _BitInt(4)), "");

  // Unary ops shouldn't go through integer promotions.
  _Static_assert(EXPR_HAS_TYPE(x4_s++, _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(++x4_s, _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(x4_u++, unsigned _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(++x4_u, unsigned _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(+x4_s, _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(-x4_s, _BitInt(4)), "");
  _Static_assert(EXPR_HAS_TYPE(~x4_u, unsigned _BitInt(4)), "");

  // This one really does convert to a different result type though.
  _Static_assert(EXPR_HAS_TYPE(!x4_u, int), "");

  // Test binary ops pick the correct common type.
  _Static_assert(EXPR_HAS_TYPE(x43_s + x_int, _BitInt(43)), "");
  _Static_assert(EXPR_HAS_TYPE(x43_u + x_int, unsigned _BitInt(43)), "");
  _Static_assert(EXPR_HAS_TYPE(x32_s + x_int, int), "");
  _Static_assert(EXPR_HAS_TYPE(x32_u + x_int, unsigned int), "");
  _Static_assert(EXPR_HAS_TYPE(x32_s + x_uint, unsigned int), "");
  _Static_assert(EXPR_HAS_TYPE(x32_u + x_uint, unsigned int), "");
  _Static_assert(EXPR_HAS_TYPE(x4_s + x_int, int), "");
  _Static_assert(EXPR_HAS_TYPE(x4_u + x_int, int), "");
  _Static_assert(EXPR_HAS_TYPE(x4_s + x_uint, unsigned int), "");
  _Static_assert(EXPR_HAS_TYPE(x4_u + x_uint, unsigned int), "");
}

void FromPaper1(void) {
  // Test the examples of conversion and promotion rules from C2x 6.3.1.8.
  _BitInt(2) a2 = 1;
  _BitInt(3) a3 = 2;
  _BitInt(33) a33 = 1;
  char c = 3;

  _Static_assert(EXPR_HAS_TYPE(a2 * a3, _BitInt(3)), "");
  _Static_assert(EXPR_HAS_TYPE(a2 * c, int), "");
  _Static_assert(EXPR_HAS_TYPE(a33 * c, _BitInt(33)), "");
}

void FromPaper2(_BitInt(8) a1, _BitInt(24) a2) {
  _Static_assert(EXPR_HAS_TYPE(a1 * (_BitInt(32))a2, _BitInt(32)), "");
}

typedef _BitInt(4) ExtVec4x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(8) ExtVec8x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(32) ExtVec32x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(128) ExtVec128x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(256) ExtVec256x2 __attribute__((ext_vector_type(2)));
typedef _BitInt(512) ExtVec512x3 __attribute__((ext_vector_type(3)));
typedef _BitInt(1024) ExtVec1024x2 __attribute__((ext_vector_type(2)));
typedef unsigned _BitInt(512) ExtUVec512x3 __attribute__((ext_vector_type(3)));
typedef _BitInt(8) GnuVec8x4 __attribute__((vector_size(4)));
typedef _BitInt(64) GnuVec64x4 __attribute__((vector_size(32)));
typedef _BitInt(256) GnuVec256x2 __attribute__((vector_size(64)));
typedef _BitInt(1024) GnuVec1024x2 __attribute__((vector_size(256)));
typedef unsigned _BitInt(256) GnuUVec256x2 __attribute__((vector_size(64)));
typedef unsigned __int128 UInt128Ext4 __attribute__((ext_vector_type(4)));
typedef unsigned __int128 UInt128Gnu2 __attribute__((vector_size(32)));
typedef __fp16 HalfExt4 __attribute__((ext_vector_type(4)));
typedef float FloatExt4 __attribute__((ext_vector_type(4)));
typedef long double LongDoubleExt4 __attribute__((ext_vector_type(4)));
typedef double DoubleGnu4 __attribute__((vector_size(32)));
typedef char CharExt4 __attribute__((ext_vector_type(4)));
typedef short ShortExt4 __attribute__((ext_vector_type(4)));
typedef int IntExt4 __attribute__((ext_vector_type(4)));
typedef __int128 Int128Ext4 __attribute__((ext_vector_type(4)));
typedef char CharGnu4 __attribute__((vector_size(4)));
typedef long long LongLongGnu4 __attribute__((vector_size(32)));
typedef __int128 Int128Gnu2 __attribute__((vector_size(32)));

void GH200665(ExtVec4x4 e4a, ExtVec4x4 e4b, ExtVec8x4 e8a, ExtVec8x4 e8b,
              ExtVec32x4 e32a, ExtVec32x4 e32b, ExtVec128x4 e128a,
              ExtVec128x4 e128b, ExtVec256x2 e256a, ExtVec256x2 e256b,
              ExtVec512x3 e512a, ExtVec512x3 e512b, ExtVec1024x2 e1024a,
              ExtVec1024x2 e1024b, ExtUVec512x3 eu512a, ExtUVec512x3 eu512b,
              GnuVec8x4 g8a, GnuVec8x4 g8b, GnuVec64x4 g64a, GnuVec64x4 g64b,
              GnuVec256x2 g256a, GnuVec256x2 g256b, GnuVec1024x2 g1024a,
              GnuVec1024x2 g1024b, GnuUVec256x2 gu256a, GnuUVec256x2 gu256b) {
  _Static_assert(EXPR_HAS_TYPE(e4a == e4b, CharExt4), "");
  _Static_assert(EXPR_HAS_TYPE(e8a != e8b, CharExt4), "");
  _Static_assert(EXPR_HAS_TYPE(e32a < e32b, IntExt4), "");
  _Static_assert(EXPR_HAS_TYPE(e128a <= e128b, Int128Ext4), "");
  _Static_assert(EXPR_HAS_TYPE(e256a > e256b, ExtVec256x2), "");
  _Static_assert(EXPR_HAS_TYPE(e512a >= e512b, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(e512a > 0, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(0 == e1024a, ExtVec1024x2), "");
  _Static_assert(EXPR_HAS_TYPE(e1024a != e1024b, ExtVec1024x2), "");
  _Static_assert(EXPR_HAS_TYPE(eu512a == eu512b, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(!e4a, CharExt4), "");
  _Static_assert(EXPR_HAS_TYPE(!e512a, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(!eu512a, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(e512a && e512b, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE(e4a || e4b, CharExt4), "");
  _Static_assert(EXPR_HAS_TYPE(e512a ? e512a : e512b, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE((e512a > 0) ? e512a : e512b, ExtVec512x3), "");
  _Static_assert(EXPR_HAS_TYPE((eu512a == eu512b) ? eu512a : eu512b, ExtUVec512x3), "");

  _Static_assert(EXPR_HAS_TYPE(g8a == g8b, CharGnu4), "");
  _Static_assert(EXPR_HAS_TYPE(g64a < g64b, LongLongGnu4), "");
  _Static_assert(EXPR_HAS_TYPE(g256a > g256b, GnuVec256x2), "");
  _Static_assert(EXPR_HAS_TYPE(g256a >= 0, GnuVec256x2), "");
  _Static_assert(EXPR_HAS_TYPE(g1024a != g1024b, GnuVec1024x2), "");
  _Static_assert(EXPR_HAS_TYPE(gu256a <= gu256b, GnuVec256x2), "");
}

void OtherElementTypes(UInt128Ext4 u128a, UInt128Ext4 u128b, UInt128Gnu2 gu128a,
                       UInt128Gnu2 gu128b, HalfExt4 ha, HalfExt4 hb, FloatExt4 fa,
                       FloatExt4 fb, LongDoubleExt4 lda, LongDoubleExt4 ldb,
                       DoubleGnu4 da, DoubleGnu4 db) {
  _Static_assert(EXPR_HAS_TYPE(u128a == u128b, Int128Ext4), "");
  _Static_assert(EXPR_HAS_TYPE(gu128a < gu128b, Int128Gnu2), "");
  _Static_assert(EXPR_HAS_TYPE(ha == hb, ShortExt4), "");
  _Static_assert(EXPR_HAS_TYPE(!ha, ShortExt4), "");
  _Static_assert(EXPR_HAS_TYPE(fa < fb, IntExt4), "");
  _Static_assert(EXPR_HAS_TYPE(fa > 0, IntExt4), "");
  _Static_assert(EXPR_HAS_TYPE(!fa, IntExt4), "");
  _Static_assert(EXPR_HAS_TYPE(fa && fb, IntExt4), "");
  _Static_assert(EXPR_HAS_TYPE((fa < fb) ? fa : fb, FloatExt4), "");
  _Static_assert(EXPR_HAS_TYPE(lda >= ldb, Int128Ext4), "");
  _Static_assert(EXPR_HAS_TYPE(da != db, LongLongGnu4), "");
}
