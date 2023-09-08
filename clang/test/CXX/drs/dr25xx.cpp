// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

namespace dr2516 { // dr2516: yes
                   // NB: reusing 1482 test
#if __cplusplus >= 201103L
template <typename T> struct S {
  typedef char I;
};
enum E2 : S<E2>::I { e };
// expected-error@-1 {{use of undeclared identifier 'E2'}}
#endif
} // namespace dr2516

namespace dr2518 { // dr2518: 17

#if __cplusplus >= 201103L
template <class T>
void f(T t) {
  if constexpr (sizeof(T) != sizeof(int)) {
#if __cplusplus < 201703L
// expected-error@-2 {{constexpr if is a C++17 extension}}
#endif
    static_assert(false, "must be int-sized"); // expected-error {{must be int-size}}
  }
}

void g(char c) {
  f(0);
  f(c); // expected-note {{requested here}}
}

template <typename Ty>
struct S {
  static_assert(false); // expected-error {{static assertion failed}}
#if __cplusplus < 201703L
// expected-error@-2 {{'static_assert' with no message is a C++17 extension}}
#endif
};

template <>
struct S<int> {};

template <>
struct S<float> {};

int test_specialization() {
  S<int> s1;
  S<float> s2;
  S<double> s3; // expected-note {{in instantiation of template class 'dr2518::S<double>' requested here}}
}
#endif

}

namespace dr2521 { // dr2521: 17
#if __cplusplus >= 201103L
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdeprecated-literal-operator"
long double operator""      _\u03C0___(long double);
// expected-warning@-1 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
// expected-warning@-2 {{user-defined literal suffixes containing '__' are reserved}}

template <char... Chars> decltype(sizeof 0)
operator""  _div();
// expected-warning@-1 {{identifier '_div' preceded by whitespace in a literal operator declaration is deprecated}}

using ::dr2521::operator"" _\u03C0___;
using ::dr2521::operator""_div;
// expected-warning@-2 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
#pragma clang diagnostic pop
#endif
} // namespace dr2521

namespace dr2565 { // dr2565: 16
#if __cplusplus >= 202002L
  template<typename T>
    concept C = requires (typename T::type x) {
      x + 1;
    };
  static_assert(!C<int>);

  // Variant of this as reported in GH57487.
  template<bool B> struct bool_constant
  { static constexpr bool value = B; };

  template<typename T>
    using is_referenceable
       = bool_constant<requires (T&) { true; }>;

  static_assert(!is_referenceable<void>::value);
  static_assert(is_referenceable<int>::value);

  template<typename T, typename U>
  concept TwoParams = requires (T *a, U b){ true;}; // #TPC

  template<typename T, typename U>
    requires TwoParams<T, U> // #TPSREQ
  struct TwoParamsStruct{};

  using TPSU = TwoParamsStruct<void, void>;
  // expected-error@-1{{constraints not satisfied for class template 'TwoParamsStruct'}}
  // expected-note@#TPSREQ{{because 'TwoParams<void, void>' evaluated to false}}
  // expected-note@#TPC{{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T, typename ...U>
  concept Variadic = requires (U* ... a, T b){ true;}; // #VC

  template<typename T, typename ...U>
    requires Variadic<T, U...> // #VSREQ
  struct VariadicStruct{};

  using VSU = VariadicStruct<void, int, char, double>;
  // expected-error@-1{{constraints not satisfied for class template 'VariadicStruct'}}
  // expected-note@#VSREQ{{because 'Variadic<void, int, char, double>' evaluated to false}}
  // expected-note@#VC{{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T>
  // expected-error@+1 {{unknown type name 'ErrorRequires'}}
  concept ErrorRequires = requires (ErrorRequires auto x) {
    x;
  };
  static_assert(ErrorRequires<int>);
  // expected-error@-1{{static assertion failed}}
  // expected-note@-2{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

  template<typename T>
  // expected-error@+2 {{unknown type name 'NestedErrorInRequires'}}
  concept NestedErrorInRequires = requires (T x) {
    requires requires (NestedErrorInRequires auto y) {
      y;
    };
  };
  static_assert(NestedErrorInRequires<int>);
  // expected-error@-1{{static assertion failed}}
  // expected-note@-2{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

#endif
}
