// RUN: %clang_cc1 -verify %s

void f0a(void) {
   inline void f1(); // expected-error {{inline declaration of 'f1' not allowed in block scope}}
}

void f0b(void) {
   void f1();
}

// FIXME: Add test for "If the inline specifier is used in a friend declaration,
// that declaration shall be a definition or the function shall have previously
// been declared inline.
