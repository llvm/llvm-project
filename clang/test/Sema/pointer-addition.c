// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

typedef struct S S; // expected-note 3 {{forward declaration of 'struct S'}}
void a(S* b, void* c) {
  void (*fp)(int) = 0;
  b++;       // expected-error {{arithmetic on a pointer to an incomplete type}}
  b += 1;    // expected-error {{arithmetic on a pointer to an incomplete type}}
  c++;       // expected-warning {{arithmetic on a pointer to void is a GNU extension}}
  c += 1;    // expected-warning {{arithmetic on a pointer to void is a GNU extension}}
  c--;       // expected-warning {{arithmetic on a pointer to void is a GNU extension}}
  c -= 1;    // expected-warning {{arithmetic on a pointer to void is a GNU extension}}
  (void) c[1]; // expected-warning {{subscript of a pointer to void is a GNU extension}}
  b = 1+b;   // expected-error {{arithmetic on a pointer to an incomplete type}}
  /* The next couple tests are only pedantic warnings in gcc */
  void (*d)(S*,void*) = a;
  d += 1;    // expected-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' is a GNU extension}}
  d++;       // expected-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' is a GNU extension}}
  d--;       // expected-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' is a GNU extension}}
  d -= 1;    // expected-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' is a GNU extension}}
  (void)(1 + d); // expected-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' is a GNU extension}}
}
