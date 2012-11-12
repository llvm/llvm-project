// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: Very incomplete!

//   A program that calls for default-initialization or value-initialization of
//   an entity of reference type is illformed. If T is a cv-qualified type, the
//   cv-unqualified version of T is used for these definitions of
//   zero-initialization, default-initialization, and value-initialization.
//
// FIXME: The diagnostics for these errors are terrible because they fall out
// of the AST representation rather than being explicitly issued during the
// respective initialization forms.
struct S { // expected-error {{implicit default constructor for 'S' must explicitly initialize the reference member}} \
           // expected-note {{candidate constructor (the implicit copy constructor) not viable}}
  int& x; // expected-note {{declared here}}
};
S s; // expected-note {{implicit default constructor for 'S' first required here}}
S f() {
  return S(); // expected-error {{no matching constructor for initialization of 'S'}}
}
