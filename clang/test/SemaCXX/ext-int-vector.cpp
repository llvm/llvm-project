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
typedef unsigned __int128 UInt128Ext4 __attribute__((ext_vector_type(4)));
typedef __fp16 HalfExt4 __attribute__((ext_vector_type(4)));
typedef _Float16 Float16Ext4 __attribute__((ext_vector_type(4)));
typedef float FloatExt4 __attribute__((ext_vector_type(4)));
typedef double DoubleExt4 __attribute__((ext_vector_type(4)));
typedef long double LongDoubleExt4 __attribute__((ext_vector_type(4)));
typedef char CharExt4 __attribute__((ext_vector_type(4)));
typedef short ShortExt4 __attribute__((ext_vector_type(4)));
typedef int IntExt4 __attribute__((ext_vector_type(4)));
typedef long LongExt4 __attribute__((ext_vector_type(4)));
typedef __int128 Int128Ext4 __attribute__((ext_vector_type(4)));

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
typedef unsigned __int128 UInt128Gnu2 __attribute__((vector_size(32)));
typedef __fp16 HalfGnu4 __attribute__((vector_size(8)));
typedef _Float16 Float16Gnu4 __attribute__((vector_size(8)));
typedef float FloatGnu4 __attribute__((vector_size(16)));
typedef double DoubleGnu4 __attribute__((vector_size(32)));
typedef long double LongDoubleGnu2 __attribute__((vector_size(32)));
typedef char CharGnu2 __attribute__((vector_size(2)));
typedef char CharGnu4 __attribute__((vector_size(4)));
typedef short ShortGnu4 __attribute__((vector_size(8)));
typedef int IntGnu4 __attribute__((vector_size(16)));
typedef long long LongLongGnu4 __attribute__((vector_size(32)));
typedef __int128 Int128Gnu2 __attribute__((vector_size(32)));

template <typename V, typename R> void check(V a, V b) {
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
  static_assert(__is_same(decltype((a < b) ? a : b), V), "");
  static_assert(__is_same(decltype(!a ? b : a), V), "");
}

template <typename V, typename R> void check_int(V a, V b) {
  check<V, R>(a, b);
  static_assert(__is_same(decltype(a ? a : b), V), "");
}

template void check_int<ExtVec2x4, CharExt4>(ExtVec2x4, ExtVec2x4);
template void check_int<ExtVec4x4, CharExt4>(ExtVec4x4, ExtVec4x4);
template void check_int<ExtVec8x4, CharExt4>(ExtVec8x4, ExtVec8x4);
template void check_int<ExtVec16x4, ShortExt4>(ExtVec16x4, ExtVec16x4);
template void check_int<ExtVec32x4, IntExt4>(ExtVec32x4, ExtVec32x4);
template void check_int<ExtVec64x4, LongExt4>(ExtVec64x4, ExtVec64x4);
template void check_int<ExtVec128x4, Int128Ext4>(ExtVec128x4, ExtVec128x4);
template void check_int<ExtVec256x2, ExtVec256x2>(ExtVec256x2, ExtVec256x2);
template void check_int<ExtVec512x3, ExtVec512x3>(ExtVec512x3, ExtVec512x3);
template void check_int<ExtVec1024x2, ExtVec1024x2>(ExtVec1024x2, ExtVec1024x2);
template void check_int<ExtUVec4x4, CharExt4>(ExtUVec4x4, ExtUVec4x4);
template void check_int<ExtUVec128x4, Int128Ext4>(ExtUVec128x4, ExtUVec128x4);
template void check_int<ExtUVec512x3, ExtVec512x3>(ExtUVec512x3, ExtUVec512x3);
template void check_int<UInt128Ext4, Int128Ext4>(UInt128Ext4, UInt128Ext4);
template void check<HalfExt4, ShortExt4>(HalfExt4, HalfExt4);
template void check<Float16Ext4, ShortExt4>(Float16Ext4, Float16Ext4);
template void check<FloatExt4, IntExt4>(FloatExt4, FloatExt4);
template void check<DoubleExt4, LongExt4>(DoubleExt4, DoubleExt4);
template void check<LongDoubleExt4, Int128Ext4>(LongDoubleExt4, LongDoubleExt4);

template void check_int<GnuVec4x2, CharGnu2>(GnuVec4x2, GnuVec4x2);
template void check_int<GnuVec8x4, CharGnu4>(GnuVec8x4, GnuVec8x4);
template void check_int<GnuVec16x4, ShortGnu4>(GnuVec16x4, GnuVec16x4);
template void check_int<GnuVec32x4, IntGnu4>(GnuVec32x4, GnuVec32x4);
template void check_int<GnuVec64x4, LongLongGnu4>(GnuVec64x4, GnuVec64x4);
template void check_int<GnuVec128x2, Int128Gnu2>(GnuVec128x2, GnuVec128x2);
template void check_int<GnuVec256x2, GnuVec256x2>(GnuVec256x2, GnuVec256x2);
template void check_int<GnuVec512x2, GnuVec512x2>(GnuVec512x2, GnuVec512x2);
template void check_int<GnuVec1024x2, GnuVec1024x2>(GnuVec1024x2, GnuVec1024x2);
template void check_int<GnuUVec256x2, GnuVec256x2>(GnuUVec256x2, GnuUVec256x2);
template void check_int<UInt128Gnu2, Int128Gnu2>(UInt128Gnu2, UInt128Gnu2);
template void check<HalfGnu4, ShortGnu4>(HalfGnu4, HalfGnu4);
template void check<Float16Gnu4, ShortGnu4>(Float16Gnu4, Float16Gnu4);
template void check<FloatGnu4, IntGnu4>(FloatGnu4, FloatGnu4);
template void check<DoubleGnu4, LongLongGnu4>(DoubleGnu4, DoubleGnu4);
template void check<LongDoubleGnu2, Int128Gnu2>(LongDoubleGnu2, LongDoubleGnu2);

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
