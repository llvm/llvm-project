// RUN: %clang_cc1 -triple loongarch64 -fsyntax-only -verify %s
typedef long long __m128i __attribute__ ((__vector_size__ (16), __may_alias__));

__m128i foo = __builtin_lsx_vinsgr2vr_w({0, 0}, 0, 0); // expected-error {{builtin needs target feature lsx}}
__m128i bar = __builtin_lsx_vfrsqrte_s({0, 0}); // expected-error {{builtin needs target feature lsx,frecipe}}