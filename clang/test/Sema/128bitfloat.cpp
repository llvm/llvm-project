// RUN: %clang_cc1 -verify -std=gnu++11 %s
// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -triple powerpc64-linux -verify -std=c++11 %s
// RUN: %clang_cc1 -triple i686-windows-gnu -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu -verify -std=c++11 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -verify -std=c++11 %s

#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)

#if defined(__ppc__)
template <typename> struct __is_float128 { static constexpr bool value = false; };
template <> struct __is_float128<__float128> { static constexpr bool value = true; };
static_assert(__is_float128<__ieee128>::value, "__ieee128 aliases to __float128");
#endif

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
