// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-gnu %s

namespace GH200665 {
typedef _BitInt(2) ExtVec2x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(4) ExtVec4x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(8) ExtVec8x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(16) ExtVec16x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(32) ExtVec32x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(64) ExtVec64x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(128) ExtVec128x4 __attribute__((ext_vector_type(4)));
typedef _BitInt(256) ExtVec256x2 __attribute__((ext_vector_type(2)));
typedef _BitInt(512) ExtVec512x3 __attribute__((ext_vector_type(3)));
typedef _BitInt(1024) ExtVec1024x2 __attribute__((ext_vector_type(2)));
typedef unsigned _BitInt(4) ExtUVec4x4 __attribute__((ext_vector_type(4)));
typedef unsigned _BitInt(128) ExtUVec128x4 __attribute__((ext_vector_type(4)));
typedef unsigned _BitInt(512) ExtUVec512x3 __attribute__((ext_vector_type(3)));

typedef _BitInt(4) GnuVec4x2 __attribute__((vector_size(2)));
typedef _BitInt(8) GnuVec8x4 __attribute__((vector_size(4)));
typedef _BitInt(16) GnuVec16x4 __attribute__((vector_size(8)));
typedef _BitInt(32) GnuVec32x4 __attribute__((vector_size(16)));
typedef _BitInt(64) GnuVec64x4 __attribute__((vector_size(32)));
typedef _BitInt(128) GnuVec128x2 __attribute__((vector_size(32)));
typedef _BitInt(256) GnuVec256x2 __attribute__((vector_size(64)));
typedef _BitInt(512) GnuVec512x2 __attribute__((vector_size(128)));
typedef _BitInt(1024) GnuVec1024x2 __attribute__((vector_size(256)));
typedef unsigned _BitInt(256) GnuUVec256x2 __attribute__((vector_size(64)));

template <typename V, typename R = V> void check(V a, V b) {
  static_assert(__is_same(decltype(a == b), R), "");
  static_assert(__is_same(decltype(a != b), R), "");
  static_assert(__is_same(decltype(a < b), R), "");
  static_assert(__is_same(decltype(a <= b), R), "");
  static_assert(__is_same(decltype(a > b), R), "");
  static_assert(__is_same(decltype(a >= b), R), "");
  static_assert(__is_same(decltype(a > 0), R), "");
  static_assert(__is_same(decltype(0 < a), R), "");
  static_assert(__is_same(decltype(!a), R), "");
  static_assert(__is_same(decltype(a && b), R), "");
  static_assert(__is_same(decltype(a || b), R), "");
  static_assert(__is_same(decltype(a ? a : b), V), "");
  static_assert(__is_same(decltype((a < b) ? a : b), V), "");
  static_assert(__is_same(decltype(!a ? b : a), V), "");
}

template void check<ExtVec2x4>(ExtVec2x4, ExtVec2x4);
template void check<ExtVec4x4>(ExtVec4x4, ExtVec4x4);
template void check<ExtVec8x4>(ExtVec8x4, ExtVec8x4);
template void check<ExtVec16x4>(ExtVec16x4, ExtVec16x4);
template void check<ExtVec32x4>(ExtVec32x4, ExtVec32x4);
template void check<ExtVec64x4>(ExtVec64x4, ExtVec64x4);
template void check<ExtVec128x4>(ExtVec128x4, ExtVec128x4);
template void check<ExtVec256x2>(ExtVec256x2, ExtVec256x2);
template void check<ExtVec512x3>(ExtVec512x3, ExtVec512x3);
template void check<ExtVec1024x2>(ExtVec1024x2, ExtVec1024x2);
template void check<ExtUVec4x4, ExtVec4x4>(ExtUVec4x4, ExtUVec4x4);
template void check<ExtUVec128x4, ExtVec128x4>(ExtUVec128x4, ExtUVec128x4);
template void check<ExtUVec512x3, ExtVec512x3>(ExtUVec512x3, ExtUVec512x3);

template void check<GnuVec4x2>(GnuVec4x2, GnuVec4x2);
template void check<GnuVec8x4>(GnuVec8x4, GnuVec8x4);
template void check<GnuVec16x4>(GnuVec16x4, GnuVec16x4);
template void check<GnuVec32x4>(GnuVec32x4, GnuVec32x4);
template void check<GnuVec64x4>(GnuVec64x4, GnuVec64x4);
template void check<GnuVec128x2>(GnuVec128x2, GnuVec128x2);
template void check<GnuVec256x2>(GnuVec256x2, GnuVec256x2);
template void check<GnuVec512x2>(GnuVec512x2, GnuVec512x2);
template void check<GnuVec1024x2>(GnuVec1024x2, GnuVec1024x2);
template void check<GnuUVec256x2, GnuVec256x2>(GnuUVec256x2, GnuUVec256x2);

typedef __int128 Int128x4 __attribute__((ext_vector_type(4)));
typedef __int128 GnuInt128x2 __attribute__((vector_size(32)));
void keeps_bitint(ExtVec128x4 a, ExtVec128x4 b, GnuVec128x2 c, GnuVec128x2 d) {
  static_assert(!__is_same(decltype(a == b), Int128x4), "");
  static_assert(!__is_same(decltype(c == d), GnuInt128x2), "");
}

// expected-error@+1{{'_BitInt' vector element width must be a power of 2}}
typedef _BitInt(7) __attribute__((ext_vector_type(4))) ExtVec7x4;
// expected-error@+1{{'_BitInt' vector element width must be a power of 2}}
typedef _BitInt(255) __attribute__((ext_vector_type(2))) ExtVec255x2;
// expected-error@+1{{'_BitInt' vector element width must be a power of 2}}
typedef _BitInt(255) __attribute__((vector_size(64))) GnuVec255x2;
} // namespace GH200665

// Original reproducer of GH200665.
// expected-error@+1{{expected ')'}}
using i512x3 = _BitInt(512) __attribute__((ext_vector_type(3));
// expected-note@-1{{'i512x3' declared here}}
// expected-error@+3{{unknown type name 'v3i512x3'; did you mean 'i512x3'?}}
// expected-warning@+2{{variable 'a' is uninitialized when used within its own initialization}}
// expected-error@+1{{expected ';' after top level declarator}}
v3i512x3 a {a>0}
