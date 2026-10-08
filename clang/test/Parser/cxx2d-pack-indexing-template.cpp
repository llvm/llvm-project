// RUN: %clang_cc1 -std=c++2d -verify -fsyntax-only %s

template <class> struct A {};
template <class> struct B {};

template <template <class> class... TT> // expected-note {{template is declared here}}
struct Malformed {
  using a = TT...[<int>;   // expected-error 2{{expected expression}} \
                           // expected-error {{expected ']'}} \
                           // expected-note {{to match this '['}} \
                           // expected-error {{expected '(' for function-style cast or type construction}}

  using b = TT...[0<int>;  // expected-error {{expected ']'}} \
                           // expected-note {{to match this '['}} \
                           // expected-error {{expected expression}} \
                           // expected-error {{expected '(' for function-style cast or type construction}}

  using c = TT...[]<int>;  // expected-error {{use of template template parameter 'TT' requires template arguments}} \
                           // expected-error {{expected ';' after alias declaration}}
};

template <template <class> class... TT>
struct S {
  using type = TT...[0]<int>;
};

template <class T> struct WithMember { using type = T; };

template <template <class> class... TT>
using U = typename TT...[0]<int>::type;

namespace ns { }
template <template <class> class... TT>
using Q = ns::TT...[0]<int>; // expected-error {{no type named 'TT' in namespace 'ns'}} \
                             // expected-error {{expected ';' after alias declaration}}

template <class T> concept Concept = true;
template <class T> constexpr int Var = 1;

template <template <class> concept... CC>
constexpr bool concept_id() {
  return CC...[0]<int>;
}

template <template <class> auto... VV>
constexpr int variable_template_id() {
  return VV...[0]<int>;
}
template <template <class> concept... CC>
struct TypeConstraint {
  template <CC...[0] T>
  static constexpr int f() { return 2; }
};
template <template <class> concept... CC>
struct MalformedConcept {
  template <CC...[] T> // expected-error {{expected identifier}}
  static void f(); // expected-error {{no candidate function template was found for dependent member function template specialization}} \
                   // expected-warning {{explicit specialization cannot have a storage class}}
};

namespace not_a_template {
template <class... T>
using U = T...[0];
static_assert(__is_same(U<int, long>, int));

template <auto... V>
constexpr auto W = V...[1];
static_assert(W<1, 2, 3> == 2);
}
