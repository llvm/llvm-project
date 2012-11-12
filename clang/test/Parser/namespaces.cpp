// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR6596
namespace g { enum { o = 0 }; }

void foo() {
  namespace a { typedef g::o o; } // expected-error{{namespaces can only be defined in global or namespace scope}}
}

// PR14085
namespace PR14085 {}
namespace = PR14085; // expected-error {{expected identifier}}
