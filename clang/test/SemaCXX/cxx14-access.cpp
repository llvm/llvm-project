// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s

namespace NoCrashOnDelayedAccessCheck {
class Foo {
  class Private; // expected-note {{declared private here}}
};

struct Bar {};

template <typename T>
Foo::Private Bar::ABC; // expected-error {{no member named 'ABC' in 'NoCrashOnDelayedAccessCheck::Bar'}} \
                          expected-error {{'Private' is a private member of}}
}
