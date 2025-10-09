// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -verify %s

@import InstancetypeModule;

void test() {
  // The nullability is here to verify that the API notes were applied.
  int good = [SomeSubclass instancetypeFactoryMethod]; // expected-error {{initializing 'int' with an expression of type 'SomeSubclass * _Nonnull'}}
  int bad = [SomeSubclass staticFactoryMethod]; // expected-error {{initializing 'int' with an expression of type 'SomeBaseClass * _Nonnull'}}
}
