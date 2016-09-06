// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

#ifdef __FLOAT128__
__float128 f;
template<typename> struct __is_floating_point_helper {};
template<> struct __is_floating_point_helper<__float128> {};
int g(int x, __float128 *y) {
  return x + *y;
}

// expected-no-diagnostics
#else
#if !defined(__STRICT_ANSI__)
__float128 f;  // expected-error {{__float128 is not supported on this target}}
// But this should work:
template<typename> struct __is_floating_point_helper {};
template<> struct __is_floating_point_helper<__float128> {};  // expected-error {{__float128 is not supported on this target}}

// FIXME: This could have a better diag.
int g(int x, __float128 *y) {  // expected-error {{__float128 is not supported on this target}}
  return x + *y;
}

#else
__float128 f;  // expected-error {{__float128 is not supported on this target}}
template<typename> struct __is_floating_point_helper {};
template<> struct __is_floating_point_helper<__float128> {};  // expected-error {{__float128 is not supported on this target}}

int g(int x, __float128 *y) {  // expected-error {{__float128 is not supported on this target}}
  return x + *y;
}

#endif
#endif
