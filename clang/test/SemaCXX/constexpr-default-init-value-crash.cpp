// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify -fno-recovery-ast

namespace NoCrash {
struct ForwardDecl; // expected-note {{forward declaration of}}
struct Foo {        // expected-note 2{{candidate constructor}}
  ForwardDecl f;    // expected-error {{field has incomplete type}}
};

constexpr Foo getFoo() {
  Foo e = 123; // expected-error {{no viable conversion from 'int' to 'Foo'}}
  return e;
}
}
