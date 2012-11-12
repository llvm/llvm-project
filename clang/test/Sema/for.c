// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check C99 6.8.5p3
void b1 (void) { for (void (*f) (void);;); }
void b2 (void) { for (void f (void);;); }   // expected-error {{declaration of non-local variable}}
void b3 (void) { for (static int f;;); }    // expected-error {{declaration of non-local variable}}
void b4 (void) { for (typedef int f;;); }   // expected-error {{declaration of non-local variable}}
