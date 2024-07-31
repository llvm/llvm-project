// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

template <typename> struct TS; // #template

// CHECK-LABEL: CXXRecordDecl {{.*}} struct Errors
struct Errors {
  // We simply ignore the '...' here.
  // CHECK: FriendDecl {{.*}} 'float'
  friend float...; // expected-error {{pack expansion does not contain any unexpanded parameter packs}}

  // CHECK-NEXT: FriendDecl {{.*}} 'short'
  // CHECK-NEXT: FriendDecl {{.*}} 'unsigned int'
  // CHECK-NEXT: FriendDecl {{.*}} 'unsigned short'
  friend short..., unsigned, unsigned short...; // expected-error 2 {{pack expansion does not contain any unexpanded parameter packs}}

  // FIXME: This is a pretty bad diagnostic.
  template <typename>
  friend struct TS, int; // expected-error {{cannot be referenced with the 'struct' specifier}}
                         // expected-note@#template {{declared here}}

  double friend; // expected-error {{'friend' must appear first in a non-function declaration}}
  double friend, double; // expected-error {{expected member name or ';' after declaration specifiers}}
};

struct C { template<class T> class Nested; }; // expected-note 2 {{'C::Nested' declared here}}
struct S {
  template<class T>
  friend class C::Nested;
};

template<class... Ts>
struct VS {
  template<class... Us>
  friend Us...; // expected-error {{friend type templates must use an elaborated type}}

  template<class... Us> // expected-note {{is declared here}}
  friend class Us...; // expected-error {{declaration of 'Us' shadows template parameter}}
                      // expected-error@-1 {{pack expansion does not contain any unexpanded parameter packs}}

  template<class U>
  friend class C<Ts>::Nested<U>...; // expected-error {{explicit specialization of non-template class 'C'}}
                                    // expected-error@-1 {{no template named 'Nested' in the global namespace}}
                                    // expected-error@-2 {{friends can only be classes or functions}}
                                    // expected-error@-3 {{expected ';' at end of declaration list}}

  template<class... Us>
  friend class C<Ts...>::Nested<Us>...; // expected-error {{explicit specialization of non-template class 'C'}}
                                        // expected-error@-1 {{no template named 'Nested' in the global namespace}}
                                        // expected-error@-2 {{friends can only be classes or functions}}
                                        // expected-error@-3 {{expected ';' at end of declaration list}}

  template<class... Us>
  friend class C<Us>::Nested...; // expected-error {{explicit specialization of non-template class 'C'}}
                                 // expected-error@-1 {{friends can only be classes or functions}}
                                 // expected-error@-2 {{expected ';' at end of declaration list}}
};


template<class... Ts> // expected-note {{template parameter is declared here}}
struct S2 {
  friend class Ts...; // expected-error {{declaration of 'Ts' shadows template parameter}}
                      // expected-error@-1 {{pack expansion does not contain any unexpanded parameter packs}}

  // TODO: Fix-it hint to insert '...'.
  friend Ts; // expected-error {{friend declaration contains unexpanded parameter pack}}
};
