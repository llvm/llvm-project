// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -fexceptions -verify %s

void func1(int i) { // expected-note{{previous definition is here}}
  int i; // expected-error{{redefinition of 'i'}}
}

void func2(int i) try { // expected-note{{previous definition is here}}
  int i; // expected-error{{redefinition of 'i'}}
} catch (...) {
}

void func3(int i) try { // expected-note {{previous definition is here}}
} catch (int i) { // expected-error {{redefinition of 'i'}}
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

void func8() {
  int i;
  try {
    int i;
  } catch (...) {
  }
}

void func9() {
  if (bool b = true)
    try {
      int b; // FIXME: this probably should be invalid, maybe
    } catch (...) {
    }
}

void func10() {
  if (bool b = true)
    if (true) {
      int b; // FIXME: decide whether this is valid
    }
}

void func11(int a) {
  try {
  } catch (int a) {  // OK
  }
}
