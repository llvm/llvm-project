// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -verify -x c++
// RUN: %clang_cc1 %s -verify -std=c17 -DC17
// RUN: %clang_cc1 %s -verify -std=c23 -DC23
// RUN: %clang_cc1 %s -verify -x c++ -std=c++26

// _BitInt is a Clang extension in C and C++ everywhere, and a standard
// feature in C23+. __has_extension(bit_int) is always 1; __has_feature
// is 1 only in C23+.

// expected-no-diagnostics

#if !__has_extension(bit_int)
#error "__has_extension(bit_int) should be true in every language mode"
#endif

#ifdef C23
#  if !__has_feature(bit_int)
#    error "__has_feature(bit_int) should be true in C23"
#  endif
#endif

#ifdef C17
#  if __has_feature(bit_int)
#    error "__has_feature(bit_int) should be false in pre-C23"
#  endif
#endif

_BitInt(13) bi;
unsigned _BitInt(48) ubi;
