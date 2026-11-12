// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// PR6596
namespace g { enum { o = 0 }; }

void foo() {
  namespace a { typedef g::o o; } // expected-error{{namespaces can only be defined in global or namespace scope}}
}

// PR14085
namespace PR14085 {}
namespace = PR14085; // expected-error {{expected identifier}}

struct namespace_nested_in_record {
  int k = ({namespace {}}); // expected-error {{statement expression not allowed at file scope}}
};
