// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

#ifndef __cpp_variadic_friend
#  error No variadic friends?
#endif

static_assert(__cpp_variadic_friend == 202403L);

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
                      // expected-error@-1 {{pack expansion does not contain any unexpanded parameter packs}}

  template<class U>
  friend class C<Ts>::template Nested<U>...; // expected-error {{cannot specialize a dependent template}}

  template<class... Us>
  friend class C<Ts...>::template Nested<Us>...; // expected-error {{cannot specialize a dependent template}}

  // FIXME: Should be valid, but we currently can’t handle packs in NNSs.
  template<class ...T>
  friend class TS<Ts>::Nested...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
                                  // expected-warning@-1 {{dependent nested name specifier 'TS<Ts>::' for friend template declaration is not supported; ignoring this friend declaration}}

  // FIXME: This I legitimately have no idea what to do with. I *think* it might
  // be well-formed by the same logic as the previous one?
  template<class T>
  friend class D<T, Ts>::Nested...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
                                    // expected-warning@-1 {{dependent nested name specifier 'D<T, Ts>::' for friend class declaration is not supported; turning off access control for 'VS'}}

  // FIXME: Ill-formed... probably?
  template<class... Us>
  friend class C<Us>::Nested...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
                                 // expected-warning@-1 {{dependent nested name specifier 'C<Us>::' for friend class declaration is not supported; turning off access control for 'VS'}}
};
