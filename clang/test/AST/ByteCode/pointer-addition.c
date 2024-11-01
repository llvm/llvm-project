// RUN: %clang_cc1 %s -fsyntax-only -verify=gnu,expected -pedantic -Wextra -std=c11 -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -triple i686-unknown-unknown -verify=gnu,expected -pedantic -Wextra -std=c11 -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -triple x86_64-unknown-unknown -verify=gnu,expected -pedantic -Wextra -std=c11 -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -Wextra -Wno-gnu -std=c11 -fexperimental-new-constant-interpreter

typedef __INTPTR_TYPE__ intptr_t;
typedef struct S S; // expected-note 4 {{forward declaration of 'struct S'}}
extern _Atomic(S*) e;
void a(S* b, void* c) {
  void (*fp)(int) = 0;
  b++;       // expected-error {{arithmetic on a pointer to an incomplete type}}
  b += 1;    // expected-error {{arithmetic on a pointer to an incomplete type}}
  c++;       // gnu-warning {{arithmetic on a pointer to void is a GNU extension}}
  c += 1;    // gnu-warning {{arithmetic on a pointer to void is a GNU extension}}
  c--;       // gnu-warning {{arithmetic on a pointer to void is a GNU extension}}
  c -= 1;    // gnu-warning {{arithmetic on a pointer to void is a GNU extension}}
  (void) c[1]; // gnu-warning {{subscript of a pointer to void is a GNU extension}}
  b = 1+b;   // expected-error {{arithmetic on a pointer to an incomplete type}}
  /* The next couple tests are only pedantic warnings in gcc */
  void (*d)(S*,void*) = a;
  d += 1;    // gnu-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' (aka 'void (struct S *, void *)') is a GNU extension}}
  d++;       // gnu-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' (aka 'void (struct S *, void *)') is a GNU extension}}
  d--;       // gnu-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' (aka 'void (struct S *, void *)') is a GNU extension}}
  d -= 1;    // gnu-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' (aka 'void (struct S *, void *)') is a GNU extension}}
  (void)(1 + d); // gnu-warning {{arithmetic on a pointer to the function type 'void (S *, void *)' (aka 'void (struct S *, void *)') is a GNU extension}}
  e++;       // expected-error {{arithmetic on a pointer to an incomplete type}}
  intptr_t i = (intptr_t)b;
  char *f = (char*)0 + i; // gnu-warning {{arithmetic on a null pointer treated as a cast from integer to pointer is a GNU extension}}
  // Cases that don't match the GNU inttoptr idiom get a different warning.
  f = (char*)0 - i; // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior}}
  int *g = (int*)0 + i; // expected-warning {{performing pointer arithmetic on a null pointer has undefined behavior}}
}
