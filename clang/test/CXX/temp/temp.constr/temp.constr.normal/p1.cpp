// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s
// RUN: %clang_cc1 -std=c++2c -x c++ -verify %s

template<typename T> concept True = true;
template<typename T> concept Foo = True<T*>; // #Foo
template<typename T> concept Bar = Foo<T&>;  // #Bar
template<typename T> requires Bar<T> struct S { }; // #S
template<typename T> requires Bar<T> && true struct S<T> { }; // #SpecS
// expected-error@-1 {{class template partial specialization is not more specialized than the primary template}}
// expected-error@#Foo 2{{'type name' declared as a pointer to a reference of type 'T &'}}
// expected-note@#SpecS {{while substituting into concept arguments here}}
// expected-note@#S {{while substituting into concept arguments here}}
// expected-note@#Bar 2{{while substituting into concept arguments here}}
// expected-note@#S {{template is declared here}}



template<typename T> concept True2 = sizeof(T) >= 0;
template<typename T> concept Foo2 = True2<T*>; // #Foo2

template<typename T> concept Bar2 = Foo2<T&>; // #Bar2
// expected-note@-1 3{{while substituting into concept arguments here; substitution failures not allowed in concept arguments}}
template<typename T> requires Bar2<T> struct S2 { }; // #SpecS2_1
// expected-note@-1{{template is declared here}}
template<typename T> requires Bar2<T> && true struct S2<T> { }; // #SpecS2_2
// expected-error@-1{{class template partial specialization is not more specialized than the primary template}}
// expected-error@#Foo2{{'type name' declared as a pointer to a reference of type 'T &'}}


namespace type_pack {
  template<typename... Args>
  concept C1 = ((sizeof(Args) >= 0) && ...);

  template<typename A, typename... B>
  concept C2 = C1<A, B...>;

  template<typename T>
  constexpr void foo() requires C2<T, char, T> { }

  template<typename T>
  constexpr void foo() requires C1<T, char, T> && true { }

  static_assert((foo<int>(), true));
}

namespace template_pack {
  template<typename T> struct S1 {};
  template<typename T> struct S2 {};

  template<template<typename> typename... Args>
  concept C1 = ((sizeof(Args<int>) >= 0) && ...);

  template<template<typename> typename A, template<typename> typename... B>
  concept C2 = C1<A, B...>;

  template<template<typename> typename T>
  constexpr void foo() requires C2<T, S1, T> { }

  template<template<typename> typename T>
  constexpr void foo() requires C1<T, S1, T> && true { }

  static_assert((foo<S2>(), true));
}

namespace non_type_pack {
  template<int... Args>
  concept C1 = ((Args >= 0) && ...);

  template<int A, int... B>
  concept C2 = C1<A, B...>;

  template<int T>
  constexpr void foo() requires C2<T, 2, T> { }

  template<int T>
  constexpr void foo() requires C1<T, 2, T> && true { }

  static_assert((foo<1>(), true));
}

namespace PR47174 {
// This checks that we don't crash with a failed substitution on the first constrained argument when
// performing normalization.
template <Bar2 T, True U> // #S3_Header
requires true struct S3; // expected-note {{template is declared here}}
template <True T, True U>
requires true struct S3<T, U>;
// expected-error@-1 {{class template partial specialization is not more specialized than the primary template}}
// expected-error@#Foo2 2{{'type name' declared as a pointer to a reference of type 'T &'}}
// expected-note@#SpecS2_1 {{while substituting into concept arguments here}}
// expected-note@#SpecS2_2 {{while substituting into concept arguments here}}
// expected-note@#S3_Header {{while substituting into concept arguments here}}
// expected-note@#Bar2 {{while substituting into concept arguments here}}


// Same as above, for the second position (but this was already working).
template <True T, Bar2 U> // #S4_Header
requires true struct S4; // #S4
template <True T, True U>
requires true struct S4<T, U>; // #S4-spec
// expected-error@-1 {{class template partial specialization is not more specialized than the primary template}}
// expected-error@#Foo2 {{'type name' declared as a pointer to a reference of type 'U &'}}
// expected-note@#S4_Header {{while substituting into concept arguments here}}
// expected-note@#S4 {{template is declared here}}
// expected-note@#S4 {{similar constraint expressions not considered equivalent}}
// expected-note@#S4-spec {{similar constraint expression here}}



struct X {
  template<int> struct Y {
    using type = int;
  };
};

template<class T> concept C1 = sizeof(T) != 0;
template<class T> concept C2 = C1<typename T::template Y<1>::type>;

template<class T> requires C1<T> void t1() {};                // expected-note {{candidate function}}
template<class T> requires C1<T> && C2<T> void t1() = delete; // expected-note {{candidate function}}
template void t1<X>();
void t1() { t1<X>(); } // expected-error {{call to deleted function 't1'}}

template<class T> requires C1<T> void t2() {}; // expected-note 2 {{candidate function}}
template<class T> requires C2<T> void t2() {}; // expected-note 2 {{candidate function}}
template void t2<X>(); // expected-error {{partial ordering for explicit instantiation of 't2' is ambiguous}}
void t2() { t2<X>(); } // expected-error {{call to 't2' is ambiguous}}
} // namespace PR47174
