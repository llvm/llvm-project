// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 -std=c++17 %s

template<template<typename T> class X> struct A; // #A
// expected-note@-1 3{{template parameter is declared here}}

template<template<typename T, int I> class X> struct B; // expected-note{{template parameter is declared here}}

template<template<int I> class X> struct C;
// expected-error@-1 {{conversion from 'int' to 'const int &' in converted constant expression would bind reference to a temporary}}
// expected-note@-2 {{template parameter is declared here}}

template<class> struct X; // expected-note {{template is declared here}}
template<int N> struct Y; // expected-note {{template parameter is declared here}}
template<long N> struct Ylong;
template<const int &N> struct Yref; // expected-note {{template parameter is declared here}}

namespace N {
  template<class> struct Z;
}
template<
  class,
  class> // expected-note {{template parameter is declared here}}
struct TooMany;


A<X> *a1;
A<N::Z> *a2;
A< ::N::Z> *a3;

A<Y> *a4; // expected-error@#A {{template argument for non-type template parameter must be an expression}}
          // expected-note@-1 {{template template argument is incompatible}}
A<TooMany> *a5; // expected-error@#A {{no template parameter in this template template parameter}}
                // expected-note@-1 {{template template argument is incompatible}}
B<X> *a6; // expected-error {{too many template arguments for class template 'X'}}
          // expected-note@-1 {{template template argument is incompatible}}
C<Y> *a7;
C<Ylong> *a8;
C<Yref> *a9; // expected-note {{template template argument is incompatible}}

template<typename T> void f(int);

A<f> *a9; // expected-error{{must be a class template}}

// Evil digraph '<:' is parsed as '[', expect error.
A<::N::Z> *a10;
#if __cplusplus <= 199711L
// expected-error@-2 {{found '<::' after a template name which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?}}
#endif

// Do not do a digraph correction here.
A<: :N::Z> *a11;  // expected-error{{expected expression}} \
                     precxx17-error{{a type specifier is required for all declarations}} \
                     cxx17-error{{expected unqualified-id}}

// PR7807
namespace N {
  template <typename, typename = int>
  struct X
  { };

  template <typename ,int>
  struct Y
  { X<int> const_ref(); };

  template <template<typename,int> class TT, typename T, int N>
  int operator<<(int, TT<T, N> a) { // expected-note{{candidate template ignored}}
    0 << a.const_ref(); // expected-error{{invalid operands to binary expression ('int' and 'X<int>')}}
  }

  void f0( Y<int,1> y){ 1 << y; } // expected-note{{in instantiation of function template specialization 'N::operator<<<N::Y, int, 1>' requested here}}
}

// PR12179
template <typename Primitive, template <Primitive...> class F>
#if __cplusplus <= 199711L
// expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif

struct unbox_args {
  typedef typename Primitive::template call<F> x;
};

template <template <typename> class... Templates>
#if __cplusplus <= 199711L
// expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif

struct template_tuple {
#if __cplusplus >= 201103L
  static constexpr int N = sizeof...(Templates);
#endif
};
template <typename T>
struct identity {};
template <template <typename> class... Templates>
#if __cplusplus <= 199711L
// expected-warning@-2 {{variadic templates are a C++11 extension}}
#endif

template_tuple<Templates...> f7() {}

#if __cplusplus >= 201103L
struct S : public template_tuple<identity, identity> {
  static_assert(N == 2, "Number of template arguments incorrect");
};
#endif

void foo() {
  f7<identity>();
}

namespace CheckDependentNonTypeParamTypes {
  template<template<typename T, typename U, T v> class X> struct A {
    // expected-note@-1 2{{template parameter is declared here}}
    void f() {
      X<int, void*, 3> x;
    }
    void g() {
      X<int, long, 3> x;
    }
    void h() {
      X<unsigned char, int, 1234> x;
      // expected-error@-1 {{evaluates to 1234, which cannot be narrowed to type 'unsigned char'}}
    }
  };

  template<typename T, typename U, U v> struct B {
    // expected-error@-1 {{conflicting deduction 'U' against 'T' for parameter}}
    static const U value = v;
  };

  // FIXME: This should probably be rejected, but the rules are at best unclear.
  A<B> ab; // expected-note {{template template argument is incompatible}}

  void use() {
    ab.f();
    ab.g();
    ab.h();
  }

  template<class> struct C {
    template<class T, T V> struct D {};
    // expected-note@-1 {{template parameter is declared here}}
    using T = D<char, 1234>;
    // expected-error@-1 {{evaluates to 1234, which cannot be narrowed to type 'char'}}
  };

  template<class T> struct E {
    template <template <T V> class TT> struct F {
      using X = TT<1234>;
    };
  };
  // FIXME: This should be rejected, as there are no valid instantiations for E<char>::F
  template struct E<char>;

#if __cplusplus >= 201703L
  template<template<auto> class TT, class V> struct G {
    using type = TT<((void)0, V::value)>;
  };
#endif
}

namespace PR32185 {
  template<template<typename T, T> class U> struct A {};
  template<template<typename T, T> class U> struct B : A<U> {};
}

namespace PR10147 {
  template<typename T> struct A {};
  template<typename T = int> struct A;
  template<template<typename...> class A> void f(A<int>*) { A<> a; } // expected-warning 0-1{{extension}}
  void g() { f((A<>*)0); }
}

#if __cplusplus >= 201703L
namespace multiple_conversions {
  constexpr int g = 1;
  struct Z {
      constexpr operator const int&() const { return g; }
      constexpr operator int() { return 2; }
  } z;

  template<template<const int&> class TT> struct A {
    static constexpr int value = TT<z>::value;
  };

  template<int I> struct B {
    static constexpr int value = I;
  };
  // FIXME: This should probably convert z to (const int &) first, then
  // convert that to int.
  static_assert(A<B>::value == 1);
  // cxx17-error@-1 {{static assertion failed}}
  // cxx17-note@-2 {{expression evaluates to '2 == 1'}}
} // namespace multiple_conversions
#endif
