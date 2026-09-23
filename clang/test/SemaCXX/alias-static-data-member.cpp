// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

int foo;
struct S {
  static const int i __attribute__((alias("foo"))) = 12; // expected-error {{definition 'i' cannot also be an alias}}
};

struct OutOfLineDefinitionWithInitializer {
  static int i __attribute__((alias("foo"))); // expected-note {{previous definition is here}}
};
int OutOfLineDefinitionWithInitializer::i = 12; // expected-error {{redefinition of 'i'}}

struct OutOfLineDefinitionWithoutInitializer {
  static int i1 __attribute__((alias("foo"))); // expected-note {{previous definition is here}}
};
int OutOfLineDefinitionWithoutInitializer::i1; // expected-error {{redefinition of 'i1'}}

struct AliasDefinition {
  static int i2 __attribute__((alias("foo")));
};
