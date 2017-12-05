// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -triple powerpc64-linux -verify %s
// RUN: %clang_cc1 -triple i686-windows-gnu -verify %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu -verify %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -verify %s

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
_Float128 f;
_Float128 tiny = __FLT128_EPSILON__;
int g(int x, _Float128 *y) {
  return x + *y;
}

// expected-no-diagnostics
#else
_Float128 f;  // expected-error {{__float128 is not supported on this target}}
float tiny = __FLT128_EPSILON__; // expected-error{{use of undeclared identifier}}
int g(int x, _Float128 *y) {  // expected-error {{__float128 is not supported on this target}}
  return x + *y;
}

#endif  // defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
