// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify

namespace dr2518 { // dr2518: 17

template <class T>
void f(T t) {
  if constexpr (sizeof(T) != sizeof(int)) {
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

}


namespace dr2565 { // dr2565: 16 open
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

}
