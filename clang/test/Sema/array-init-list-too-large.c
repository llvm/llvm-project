// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i386 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple msp430 -fsyntax-only -verify=expected,size16 %s
// RUN: %clang_cc1 -triple avr -fsyntax-only -verify=expected,size16 %s
// RUN: %clang_cc1 -triple msp430 -fsyntax-only -verify=expected,size16 -x c++ %s

#if __SIZEOF_SIZE_T__ == 8
// Array sizes are limited to 61 bits.
typedef char Quarter[1ULL << 59];
#else
typedef char Quarter[__SIZE_MAX__ / 4 + 1];
#endif
Quarter fits[] = {{0}, {0}, {0}};
Quarter explicit_bound[4]; // expected-error {{array is too large (4 elements)}}
Quarter deduced_bound[] = {{0}, {0}, {0}, {0}}; // expected-error {{array is too large (4 elements)}}

#define X2(x) x, x
#define X4(x) X2(x), X2(x)
#define X8(x) X4(x), X4(x)
#define X16(x) X8(x), X8(x)
#define X32(x) X16(x), X16(x)
#define X64(x) X32(x), X32(x)
#define X128(x) X64(x), X64(x)
#define X256(x) X128(x), X128(x)
#define X512(x) X256(x), X256(x)
#define X1024(x) X512(x), X512(x)
#define X2048(x) X1024(x), X1024(x)
#define X4096(x) X2048(x), X2048(x)
#define X8192(x) X4096(x), X4096(x)
#define X16384(x) X8192(x), X8192(x)
#define X32768(x) X16384(x), X16384(x)

char past_int16_max[] = {X32768(0), 0};
_Static_assert(sizeof(past_int16_max) == 32769, "");

char past_size16_max[] = {X32768(0), X32768(0)}; // size16-error {{array is too large (65'536 elements)}}

struct E {};
struct E zero_sized[] = {X32768({}), X32768({})}; // size16-error {{array is too large (65'536 elements)}}

#ifdef __cplusplus
template <typename T> void deduced_in_template() {
  static T a[] = {{0}, {0}, {0}, {0}}; // expected-error {{array is too large (4 elements)}}
}
template void deduced_in_template<Quarter>(); // expected-note {{in instantiation of}}
#endif
