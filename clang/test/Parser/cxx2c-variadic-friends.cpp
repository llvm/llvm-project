// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

template <typename> struct TS; // #template

struct Errors {
  friend int, int;
  friend int, long, char;

  // We simply ignore the '...' here.
  friend float...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}

  friend short..., unsigned, unsigned short...; // expected-error 2 {{pack expansion does not contain any unexpanded parameter packs}}

  // FIXME: This is a pretty bad diagnostic.
  template <typename>
  friend struct TS, int; // expected-error {{cannot be referenced with the 'struct' specifier}}
                         // expected-note@#template {{declared here}}

  double friend; // expected-error {{'friend' must appear first in a non-function declaration}}
  double friend, double; // expected-error {{expected member name or ';' after declaration specifiers}}
};

template <typename>
struct C { template<class T> class Nested; };

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

  // FIXME: Ill-formed.
  template<class U>
  friend class C<Ts>::template Nested<U>...;

  // FIXME: Ill-formed.
  template<class... Us>
  friend class C<Ts...>::template Nested<Us>...;

  // FIXME: Ill-formed.
  template<class... Us>
  friend class C<Us>::Nested...;
};
