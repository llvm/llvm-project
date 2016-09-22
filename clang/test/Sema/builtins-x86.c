// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s

typedef long long __m128i __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));
typedef double __m128d __attribute__((__vector_size__(16)));

typedef float __m512 __attribute__((__vector_size__(64)));
typedef double __m512d __attribute__((__vector_size__(64)));

typedef unsigned char __mmask8;
typedef unsigned short __mmask16;

__m128 test__builtin_ia32_cmpps(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpps(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmppd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmppd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128 test__builtin_ia32_cmpss(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpss(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmpsd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmpsd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__mmask16 test__builtin_ia32_cmpps512_mask(__m512d __a, __m512d __b) {
  __builtin_ia32_cmpps512_mask(__a, __b, 32, -1, 4); // expected-error {{argument should be a value from 0 to 31}}
}

__mmask8 test__builtin_ia32_cmppd512_mask(__m512d __a, __m512d __b) {
  __builtin_ia32_cmppd512_mask(__a, __b, 32, -1, 4); // expected-error {{argument should be a value from 0 to 31}}
}

__m128i test__builtin_ia32_vpcomub(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomub(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomuw(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuw(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomud(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomud(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomuq(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuq(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomb(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomub(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomw(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuw(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomd(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomud(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomq(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuq(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}
