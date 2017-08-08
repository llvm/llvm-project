// RUN: %clang_cc1 -fsyntax-only -verify %s

void f0() { // expected-note {{previous definition is here}}
}

inline void f0(); // expected-error {{inline declaration of 'f0' follows non-inline definition}}

void func_01() {} // expected-note{{previous definition is here}}
struct C01 {
  friend void func_01() {} // expected-error{{redefinition of 'func_01'}}
};

void func_02() {} // expected-note{{previous definition is here}}
struct C02 {
  friend inline void func_02(); // expected-error{{inline declaration of 'func_02' follows non-inline definition}}
};

void func_03() {} // expected-note{{previous definition is here}}
struct C03 {
  friend inline void func_03() {} // expected-error{{inline declaration of 'func_03' follows non-inline definition}}
};

void func_04() {} // expected-note{{previous definition is here}}
inline void func_04() {} // expected-error{{inline declaration of 'func_04' follows non-inline definition}}

void func_06() {} // expected-note{{previous definition is here}}
template<typename T> struct C06 {
  friend inline void func_06() {} // expected-error{{inline declaration of 'func_06' follows non-inline definition}}
};

void func_07() {} // expected-note{{previous definition is here}}
template<typename T> struct C07 {
  friend inline void func_07(); // expected-error{{inline declaration of 'func_07' follows non-inline definition}}
};
