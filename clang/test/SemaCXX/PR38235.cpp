// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

enum class E { Foo, Bar = 97119 };

void f() __attribute__((constructor(E::Foo))); // expected-error{{'constructor' attribute requires an integer constant}}
void f2() __attribute__((constructor(E::Bar)));// expected-error{{'constructor' attribute requires an integer constant}}

void switch_me(E e) {
  switch (e) {
    case E::Foo:
    case E::Bar:
      break;
  }
}

enum class E2;

struct S {
  static const E e = E::Foo;
};
