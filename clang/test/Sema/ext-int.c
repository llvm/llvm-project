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
