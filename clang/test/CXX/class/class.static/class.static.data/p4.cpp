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
  static const int i = 0; // expected-note{{previous definition is here}}
};
int const InClassInitializerAndOutOfClassCopyInitializer::i = 0; // expected-error{{redefinition of 'i'}}

struct InClassInitializerAndOutOfClassDirectInitializer {
  static const int i = 0; // expected-note{{previous definition is here}}
};
int const InClassInitializerAndOutOfClassDirectInitializer::i(0); // expected-error{{redefinition of 'i'}}



int main() { }

