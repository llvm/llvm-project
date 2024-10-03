// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23 -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
struct type_info{};
} // namespace std

// cwg2504 is in cwg2504.cpp

namespace cwg2512 { // cwg2512: 2.7
struct A; // #cwg2512-A
void foo(A* p) {
  typeid(*p);
  // expected-error@-1 {{'typeid' of incomplete type 'A'}}
  //   expected-note@#cwg2512-A {{forward declaration of 'cwg2512::A'}}
}
} // namespace cwg2512

namespace cwg2516 { // cwg2516: 3.0
                   // NB: reusing 1482 test
#if __cplusplus >= 201103L
template <typename T> struct S {
  typedef char I;
};
enum E2 : S<E2>::I { e };
// since-cxx11-error@-1 {{use of undeclared identifier 'E2'}}
#endif
} // namespace cwg2516

namespace cwg2518 { // cwg2518: 17

#if __cplusplus >= 201103L
template <class T>
void f(T t) {
  if constexpr (sizeof(T) != sizeof(int)) {
  // cxx11-14-error@-1 {{constexpr if is a C++17 extension}}
    static_assert(false, "must be int-sized");
    // since-cxx11-error@-1 {{static assertion failed: must be int-sized}}
    //   since-cxx11-note@#cwg2518-f-c {{in instantiation of function template specialization 'cwg2518::f<char>' requested here}}
  }
}

void g(char c) {
  f(0);
  f(c); // #cwg2518-f-c
}

template <typename Ty>
struct S {
  static_assert(false);
  // cxx11-14-error@-1 {{'static_assert' with no message is a C++17 extension}}
  // since-cxx11-error@-2 {{static assertion failed}}
  //   since-cxx11-note@#cwg2518-S-double {{in instantiation of template class 'cwg2518::S<double>' requested here}}
};

template <>
struct S<int> {};

template <>
struct S<float> {};

int test_specialization() {
  S<int> s1;
  S<float> s2;
  S<double> s3; // #cwg2518-S-double
}
#endif

}

namespace cwg2521 { // cwg2521: 17
#if __cplusplus >= 201103L
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdeprecated-literal-operator"
long double operator""      _\u03C0___(long double);
// since-cxx11-warning@-1 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
// since-cxx11-warning@-2 {{user-defined literal suffixes containing '__' are reserved}}

template <char... Chars> decltype(sizeof 0)
operator""  _div();
// since-cxx11-warning@-1 {{identifier '_div' preceded by whitespace in a literal operator declaration is deprecated}}

using ::cwg2521::operator"" _\u03C0___;
using ::cwg2521::operator""_div;
// since-cxx11-warning@-2 {{identifier '_π___' preceded by whitespace in a literal operator declaration is deprecated}}
#pragma clang diagnostic pop
#endif
} // namespace cwg2521

namespace cwg2547 { // cwg2547: 20
#if __cplusplus >= 202302L
struct S;
// since-cxx23-note@-1 {{forward declaration of 'cwg2547::S'}}
// since-cxx23-note@-2 {{forward declaration of 'cwg2547::S'}}
// since-cxx23-note@-3 {{forward declaration of 'cwg2547::S'}}
bool operator==(S, S) = default;  // error: S is not complete
// since-cxx23-error@-1 {{variable has incomplete type 'S'}}
// since-cxx23-error@-2 {{variable has incomplete type 'S'}}
// since-cxx23-error@-3 {{equality comparison operator is not a friend of incomplete class 'cwg2547::S'}}
struct S {
  friend bool operator==(S, const S&) = default; // error: parameters of different types
  // since-cxx23-error@-1 {{parameters for defaulted equality comparison operator must have the same type (found 'S' vs 'const S &')}}
};
enum E { };
bool operator==(E, E) = default;  // error: not a member or friend of a class
// since-cxx23-error@-1 {{invalid parameter type for non-member defaulted equality comparison operator; found 'E', expected class or reference to a constant class}}

struct S2 {
  bool operator==(this int, S2) = default;
  // since-cxx23-error@-1 {{invalid parameter type for defaulted equality comparison operator; found 'int', expected 'const cwg2547::S2 &'}}
};
#endif
} // namespace cwg2547

#if __cplusplus >= 202302L
namespace cwg2553 { // cwg2553: 18 review 2023-07-14
struct B {
  virtual void f(this B&); 
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  static void f(this B&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a static function}}
  virtual void g(); // #cwg2553-g
};
struct D : B {
  void g(this D&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#cwg2553-g {{overridden virtual function is here}}
};

}
#endif

#if __cplusplus >= 202302L
namespace cwg2554 { // cwg2554: 18 review 2021-12-10
struct B {
  virtual void f(); // #cwg2554-g
};

struct D : B {
  void f(this D&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#cwg2554-g {{overridden virtual function is here}}
};

struct D2 : B {
  void f(this B&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#cwg2554-g {{overridden virtual function is here}}
};
struct T {};
struct D3 : B {
  void f(this T&);
  // since-cxx23-error@-1 {{an explicit object parameter cannot appear in a virtual function}}
  //   since-cxx23-note@#cwg2554-g {{overridden virtual function is here}}
};

}
#endif

#if __cplusplus >= 202302L
namespace cwg2561 { // cwg2561: no
struct C {
    constexpr C(auto) { }
};
void foo() {
    constexpr auto b = [](this C) { return 1; };
    // FIXME: closure type shouldn't have a conversion function to function
    //        pointer, because explicit object parameter is present. 
    constexpr int (*fp)(C) = b;
    static_assert(fp(1) == 1);
    static_assert((&decltype(b)::operator())(1) == 1);
}

}
#endif


namespace cwg2565 { // cwg2565: 16 open 2023-06-07
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
  concept TwoParams = requires (T *a, U b){ true;}; // #cwg2565-TPC

  template<typename T, typename U>
    requires TwoParams<T, U> // #cwg2565-TPSREQ
  struct TwoParamsStruct{};

  using TPSU = TwoParamsStruct<void, void>;
  // since-cxx20-error@-1 {{constraints not satisfied for class template 'TwoParamsStruct'}}
  //   since-cxx20-note@#cwg2565-TPSREQ {{because 'TwoParams<void, void>' evaluated to false}}
  //   since-cxx20-note@#cwg2565-TPC {{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T, typename ...U>
  concept Variadic = requires (U* ... a, T b){ true;}; // #cwg2565-VC

  template<typename T, typename ...U>
    requires Variadic<T, U...> // #cwg2565-VSREQ
  struct VariadicStruct{};

  using VSU = VariadicStruct<void, int, char, double>;
  // since-cxx20-error@-1 {{constraints not satisfied for class template 'VariadicStruct'}}
  //   since-cxx20-note@#cwg2565-VSREQ {{because 'Variadic<void, int, char, double>' evaluated to false}}
  //   since-cxx20-note@#cwg2565-VC {{because 'b' would be invalid: argument may not have 'void' type}}

  template<typename T>
  concept ErrorRequires = requires (ErrorRequires auto x) {
  // since-cxx20-error@-1 {{a concept definition cannot refer to itself}} \
  // since-cxx20-error@-1 {{'auto' not allowed in requires expression parameter}} \
  // since-cxx20-note@-1 {{declared here}}
    x;
  };
  static_assert(ErrorRequires<int>);
  // since-cxx20-error@-1 {{static assertion failed}}
  //   since-cxx20-note@-2 {{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

  template<typename T>
  concept NestedErrorInRequires = requires (T x) { //
    // since-cxx20-note@-1 {{declared here}}
    requires requires (NestedErrorInRequires auto y) {
    // since-cxx20-error@-1 {{a concept definition cannot refer to itself}} \
    // since-cxx20-error@-1 {{'auto' not allowed in requires expression parameter}}
      y;
    };
  };
  static_assert(NestedErrorInRequires<int>);
  // expected-error@-1 {{static assertion failed}}
  //   expected-note@-2 {{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

#endif
}

namespace cwg2583 { // cwg2583: 19
#if __cplusplus >= 201103L
struct A {
  int i;
  char c;
};

struct B {
  int i;
  alignas(8) char c;
};

union U {
  A a;
  B b;
};

union V {
  A a;
  alignas(64) B b;
};

static_assert(!__is_layout_compatible(A, B), "");
static_assert(__is_layout_compatible(U, V), "");
#endif
} // namespace cwg2583

namespace cwg2586 { // cwg2586: 20
#if __cplusplus >= 202302L
struct X {
  X& operator=(this X&, const X&) = default;
  X& operator=(this X&, X&) = default;
  X& operator=(this X&&, X&&) = default;
  // FIXME: The notes could be clearer on *how* the type differs
  // e.g., "if an explicit object parameter is used it must be of type reference to 'X'"
  X& operator=(this int, const X&) = default;
  // since-cxx23-warning@-1 {{explicitly defaulted copy assignment operator is implicitly deleted}}
  // since-cxx23-note@-2 {{function is implicitly deleted because its declared type does not match the type of an implicit copy assignment operator}}
  X& operator=(this X, const X&) = default;
  // since-cxx23-warning@-1 {{explicitly defaulted copy assignment operator is implicitly deleted}}
  // since-cxx23-note@-2 {{function is implicitly deleted because its declared type does not match the type of an implicit copy assignment operator}}
};
struct Y {
  void operator=(this int, const Y&); // This is copy constructor, suppresses implicit declaration
};
static_assert([]<typename T = Y>{
  return !requires(T t, const T& ct) { t = ct; };
}());

struct Z {
  bool operator==(this const Z&, const Z&) = default;
  bool operator==(this Z, Z) = default;
  bool operator==(this Z, const Z&) = default;
  // since-cxx23-error@-1 {{parameters for defaulted equality comparison operator must have the same type (found 'Z' vs 'const Z &')}}
  bool operator==(this const Z&, Z) = default;
  // since-cxx23-error@-1 {{parameters for defaulted equality comparison operator must have the same type (found 'const Z &' vs 'Z')}}
  bool operator==(this int, Z) = default;
  // since-cxx23-error@-1 {{invalid parameter type for defaulted equality comparison operator; found 'int', expected 'const cwg2586::Z &'}}
};
#endif
} // namespace cwg2586

namespace cwg2598 { // cwg2598: 18
#if __cplusplus >= 201103L
struct NonLiteral {
    NonLiteral();
};

struct anonymous1 {
    union {} a;
};
static_assert(__is_literal(anonymous1), "");

struct anonymous2 {
    union { char c; };
};
static_assert(__is_literal(anonymous2), "");

struct anonymous3 {
    union { char c; NonLiteral NL; };
};
static_assert(__is_literal(anonymous3), "");

struct anonymous4 {
    union { NonLiteral NL; };
};
static_assert(!__is_literal(anonymous4), "");

union empty {};
static_assert(__is_literal(empty), "");

union union1 { char c; };
static_assert(__is_literal(union1), "");

union union2 { char c; NonLiteral NL;};
static_assert(__is_literal(union2), "");

union union3 { NonLiteral NL;};
static_assert(!__is_literal(union3), "");

union union4 { union4(); };
static_assert(!__is_literal(union4), "");

union union5 { static NonLiteral NL; };
static_assert(__is_literal(union5), "");

struct Literal { constexpr Literal() {} };
union union6 { NonLiteral NL; Literal L; };
static_assert(__is_literal(union6), "");

#if __cplusplus >= 202003L
struct A { A(); };
union U {
  A a;
  constexpr U() {}
  constexpr ~U() {}
};
static_assert(!__is_literal(U), "");
#endif



#endif
}
