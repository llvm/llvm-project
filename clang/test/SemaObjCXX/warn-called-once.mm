// RUN: %clang_cc1 -verify -fsyntax-only -Wcompletion-handler %s

// expected-no-diagnostics

class HasCtor {
  HasCtor(void *) {}
};
