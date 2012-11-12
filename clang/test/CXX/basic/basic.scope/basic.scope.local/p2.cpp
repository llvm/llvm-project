// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -fexceptions -verify %s

void func1(int i) { // expected-note{{previous definition is here}}
  int i; // expected-error{{redefinition of 'i'}}
}

void func2(int i) try { // expected-note{{previous definition is here}}
  int i; // expected-error{{redefinition of 'i'}}
} catch (...) {
}

void func3(int i) try { // FIXME: note {{previous definition is here}}
} catch (int i) { // FIXME: error {{redefinition of 'i'}}
}

void func4(int i) try { // expected-note{{previous definition is here}}
} catch (...) {
  int i; // expected-error{{redefinition of 'i'}}
}

void func5() try {
  int i;
} catch (...) {
  int j = i; // expected-error{{use of undeclared identifier 'i'}}
}

void func6() try {
} catch (int i) { // expected-note{{previous definition is here}}
  int i; // expected-error{{redefinition of 'i'}}
}

void func7() {
  try {
  } catch (int i) { // expected-note{{previous definition is here}}
    int i; // expected-error{{redefinition of 'i'}}
  }
}
