// RUN: %clang_cc1 -fsyntax-only -verify %s 
struct InClassInitializerOnly {
  static const int i = 0;
};
int const InClassInitializerOnly::i;

struct OutOfClassInitializerOnly {
  static const int i;
};
int const OutOfClassInitializerOnly::i = 0;

struct InClassInitializerAndOutOfClassCopyInitializer {
  static const int i = 0; // expected-note{{previous initialization is here}}
};
int const InClassInitializerAndOutOfClassCopyInitializer::i = 0; // expected-error{{static data member 'i' already has an initializer}}

struct InClassInitializerAndOutOfClassDirectInitializer {
  static const int i = 0; // expected-note{{previous initialization is here}}
};
int const InClassInitializerAndOutOfClassDirectInitializer::i(0); // expected-error{{static data member 'i' already has an initializer}}


int main() { }

