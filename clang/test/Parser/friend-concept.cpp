// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

template<class>
concept fooable = true;

struct S {
  template<class> friend concept x = requires { requires true; }; // expected-error {{friend declaration cannot be a concept}}
  template<class> friend concept fooable; // expected-error {{friend declaration cannot be a concept}}
  template<class> concept friend fooable; // expected-error {{expected unqualified-id}}
  friend concept fooable; // expected-error {{friend declaration cannot be a concept}}
  concept friend fooable; // expected-error {{friend declaration cannot be a concept}}
  concept fooable; // expected-error {{concept declarations may only appear in global or namespace scope}}
};
