// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Classes.
namespace Class {
  namespace NS {
    class C {}; // expected-note {{candidate}}
  }
  using namespace NS;
  class C : C {}; // expected-error {{reference to 'C' is ambiguous}} \
                     expected-note {{candidate}}
}

// Enumerations.
enum E {
  EPtrSize = sizeof((E*)0) // ok, E is already declared
};

// Alias declarations. clang implements the proposed resolution to N1044.
namespace Alias {
  namespace NS {
    class C;
  }
  using namespace NS;
  using C = C; // ok, C = B::C
  using C = NS::C; // ok, same type
}
