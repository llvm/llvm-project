// RUN: %clang_cc1 -triple x86_64-linux-gnu -fms-compatibility -fms-extensions -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-compatibility -fms-extensions -fsyntax-only -verify -std=c++11 -DMS %s

template <bool> struct enable_if {};
template<> struct enable_if<true> { typedef void type; };

template <typename, typename> struct is_same { static constexpr bool value = false; };
template <typename T> struct is_same<T, T> { static constexpr bool value = true; };




struct S {
  // The only numeric types S can be converted to is __int128 and __float128.
  template <typename T, typename = typename enable_if<
                            !((__is_integral(T) && sizeof(T) != 16) ||
                              is_same<T, float>::value ||
                              is_same<T, double>::value ||
                              is_same<T, long double>::value)>::type>
  operator T() { return T(); }
};

void f() {
#ifdef MS
  // When targeting Win32, __float128 and __int128 do not exist, so the S
  // object cannot be converted to anything usable in the expression.
  // expected-error@+2{{invalid operands to binary expression ('S' and 'double')}}
#endif
  double d = S() + 1.0;
#ifndef MS
  // expected-error@-2{{use of overloaded operator '+' is ambiguous}}
  // expected-note@-3 36{{built-in candidate operator+}}
#endif
}
