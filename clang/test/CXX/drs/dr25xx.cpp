// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
// expected-no-diagnostics
#endif

namespace dr2516 { // dr2516: 3.0
                   // NB: reusing 1482 test
#if __cplusplus >= 201103L
template <typename T> struct S {
  typedef char I;
};
enum E2 : S<E2>::I { e };
// since-cxx11-error@-1 {{use of undeclared identifier 'E2'}}
#endif
} // namespace dr2516

namespace dr2518 { // dr2518: 17

#if __cplusplus >= 201103L
template <class T>
void f(T t) {
  if constexpr (sizeof(T) != sizeof(int)) {
  // cxx11-14-error@-1 {{constexpr if is a C++17 extension}}
    static_assert(false, "must be int-sized");
    // since-cxx11-error@-1 {{static assertion failed: must be int-sized}}
    //   since-cxx11-note@#dr2518-f-c {{in instantiation of function template specialization 'dr2518::f<char>' requested here}}
  }
}

void g(char c) {
  f(0);
  f(c); // #dr2518-f-c
}

template <typename Ty>
struct S {
  static_assert(false);
  // cxx11-14-error@-1 {{'static_assert' with no message is a C++17 extension}}
  // since-cxx11-error@-2 {{static assertion failed}}
  //   since-cxx11-note@#dr2518-S-double {{in instantiation of template class 'dr2518::S<double>' requested here}}
};

template <>
struct S<int> {};

template <>
struct S<float> {};

int test_specialization() {
  S<int> s1;
  S<float> s2;
  S<double> s3; // #dr2518-S-double
}
#endif

}

namespace dr2521 { // dr2521: 17
#if __cplusplus >= 201103L
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdeprecated-literal-operator"
long double operator""      _\u03C0___(long double);
// since-cxx11-warning@-1 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
// since-cxx11-warning@-2 {{user-defined literal suffixes containing '__' are reserved}}

template <char... Chars> decltype(sizeof 0)
operator""  _div();
// since-cxx11-warning@-1 {{identifier '_div' preceded by whitespace in a literal operator declaration is deprecated}}

using ::dr2521::operator"" _\u03C0___;
using ::dr2521::operator""_div;
// since-cxx11-warning@-2 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
#pragma clang diagnostic pop
#endif
} // namespace dr2521


#if __cplusplus >= 202302L
namespace dr2553 { // dr2553: 18
struct B {
  virtual void f(this B&); 
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  static void f(this B&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a static function}}
  virtual void g(); // #dr2553-g
};
struct D : B {
  void g(this D&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#dr2553-g {{overridden virtual function is here}}
};

}
#endif

#if __cplusplus >= 202302L
namespace dr2554 { // dr2554: 18 review
struct B {
  virtual void f(); // #dr2554-g
};

struct D : B {
  void f(this D&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#dr2554-g {{overridden virtual function is here}}
};

struct D2 : B {
  void f(this B&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#dr2554-g {{overridden virtual function is here}}
};
struct T {};
struct D3 : B {
  void f(this T&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#dr2554-g {{overridden virtual function is here}}
};

}
#endif

#if __cplusplus >= 202302L
namespace dr2561 { // dr2561: 18 review
struct C {
    constexpr C(auto) { }
};
void foo() {
    constexpr auto b = [](this C) { return 1; };
    constexpr int (*fp)(C) = b;
    static_assert(fp(1) == 1);
    static_assert((&decltype(b)::operator())(1) == 1);
}

}
#endif


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
  concept TwoParams = requires (T *a, U b){ true;}; // #dr2565-TPC

  template<typename T, typename U>
    requires TwoParams<T, U> // #dr2565-TPSREQ
  struct TwoParamsStruct{};

  using TPSU = TwoParamsStruct<void, void>;
  // since-cxx20-error@-1 {{constraints not satisfied for class template 'TwoParamsStruct'}}
  //   since-cxx20-note@#dr2565-TPSREQ {{because 'TwoParams<void, void>' evaluated to false}}
  //   since-cxx20-note@#dr2565-TPC {{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T, typename ...U>
  concept Variadic = requires (U* ... a, T b){ true;}; // #dr2565-VC

  template<typename T, typename ...U>
    requires Variadic<T, U...> // #dr2565-VSREQ
  struct VariadicStruct{};

  using VSU = VariadicStruct<void, int, char, double>;
  // since-cxx20-error@-1 {{constraints not satisfied for class template 'VariadicStruct'}}
  //   since-cxx20-note@#dr2565-VSREQ {{because 'Variadic<void, int, char, double>' evaluated to false}}
  //   since-cxx20-note@#dr2565-VC {{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T>
  concept ErrorRequires = requires (ErrorRequires auto x) {
  // since-cxx20-error@-1 {{unknown type name 'ErrorRequires'}}
    x;
  };
  static_assert(ErrorRequires<int>);
  // since-cxx20-error@-1 {{static assertion failed}}
  //   since-cxx20-note@-2 {{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

  template<typename T>
  concept NestedErrorInRequires = requires (T x) {
    requires requires (NestedErrorInRequires auto y) {
    // since-cxx20-error@-1 {{unknown type name 'NestedErrorInRequires'}}
      y;
    };
  };
  static_assert(NestedErrorInRequires<int>);
  // expected-error@-1 {{static assertion failed}}
  //   expected-note@-2 {{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

#endif
}
