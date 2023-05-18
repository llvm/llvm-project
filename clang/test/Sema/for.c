// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check C99 6.8.5p3
void b1 (void) { for (void (*f) (void);;); }
void b2 (void) { for (void f (void);;); }   // expected-error {{non-variable declaration in 'for' loop}}
void b3 (void) { for (static int f;;); }    // expected-error {{declaration of non-local variable}}
void b4 (void) { for (typedef int f;;); }   // expected-error {{non-variable declaration in 'for' loop}}
void b5 (void) { for (struct { int i; } s;;); }
void b6 (void) { for (enum { zero, ten = 10 } i;;); }
void b7 (void) { for (struct s { int i; };;); } // expected-error {{non-variable declaration in 'for' loop}}
void b8 (void) { for (static struct { int i; } s;;); } // expected-error {{declaration of non-local variable}}
void b9 (void) { for (struct { int i; } (*s)(struct { int j; } o) = 0;;); }
void b10(void) { for (typedef struct { int i; } (*s)(struct { int j; });;); } // expected-error {{non-variable declaration in 'for' loop}}
