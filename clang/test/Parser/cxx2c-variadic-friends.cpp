// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

template <typename> struct TS; // #template

struct Errors {
  friend int, int;
  friend int, long, char;

  // We simply diagnose and ignore the '...' here.
  friend float...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}

  friend short..., unsigned, unsigned short...; // expected-error 2 {{pack expansion does not contain any unexpanded parameter packs}}

  template <typename>
  friend struct TS, int; // expected-error {{a friend declaration that befriends a template must contain exactly one type-specifier}}

  double friend; // expected-error {{'friend' must appear first in a non-function declaration}}
  double friend, double; // expected-error {{expected member name or ';' after declaration specifiers}}
};

template <typename>
struct C { template<class T> class Nested; };

template <typename, typename>
struct D { template<class T> class Nested; };

template <bool>
struct E { template<class T> class Nested; };

template<class... Ts> // expected-note {{template parameter is declared here}}
struct VS {
  friend Ts...;

  friend class Ts...; // expected-error {{declaration of 'Ts' shadows template parameter}}
  // expected-error@-1 {{pack expansion does not contain any unexpanded parameter packs}}

  // TODO: Fix-it hint to insert '...'.
  friend Ts; // expected-error {{friend declaration contains unexpanded parameter pack}}

  template<class... Us>
  friend Us...; // expected-error {{friend type templates must use an elaborated type}}

  template<class... Us> // expected-note {{is declared here}}
  friend class Us...; // expected-error {{declaration of 'Us' shadows template parameter}}

  template<class U>
  friend class C<Ts>::template Nested<U>...; // expected-error {{cannot specialize a dependent template}}

  template<class... Us>
  friend class C<Ts...>::template Nested<Us>...; // expected-error {{cannot specialize a dependent template}}

  // Nonsense (see CWG 2917).
  template<class... Us>
  friend class C<Us>::Nested...; // expected-error {{friend declaration expands pack 'Us' that is declared it its own template parameter list}}

  template<bool... Bs>
  friend class E<Bs>::Nested...; // expected-error {{friend declaration expands pack 'Bs' that is declared it its own template parameter list}}

  // FIXME: Both of these should be valid, but we can't handle these at
  // the moment because the NNS is dependent.
  template<class ...T>
  friend class TS<Ts>::Nested...; // expected-warning {{dependent nested name specifier 'TS<Ts>::' for friend template declaration is not supported; ignoring this friend declaration}}

  template<class T>
  friend class D<T, Ts>::Nested...; // expected-warning {{dependent nested name specifier 'D<T, Ts>::' for friend class declaration is not supported; turning off access control for 'VS'}}
};

namespace length_mismatch {
struct A {
  template <typename...>
  struct Nested {
    struct Foo{};
  };
};
template <typename ...Ts>
struct S {
  template <typename ...Us>
  struct T {
    // expected-error@+2 {{pack expansion contains parameter packs 'Ts' and 'Us' that have different lengths (1 vs. 2)}}
    // expected-error@+1 {{pack expansion contains parameter packs 'Ts' and 'Us' that have different lengths (2 vs. 1)}}
    friend class Ts::template Nested<Us>::Foo...;
  };
};

void f() {
  S<A>::T<int> s;
  S<A, A>::T<int, long> s2;
  S<A>::T<int, long> s3; // expected-note {{in instantiation of}}
  S<A, A>::T<int> s4; // expected-note {{in instantiation of}}
}
}
