// RUN: %clang_cc1 -fsyntax-only -verify %s -triple i386-pc-unknown
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-apple-darwin9
// RUN: %clang_cc1 -fsyntax-only -fms-compatibility -DMS -verify %s -triple x86_64-pc-win32

void f1(int a)
{
    __builtin_va_list ap;

    __builtin_va_start(ap, a, a); // expected-error {{too many arguments to function}}
    __builtin_va_start(ap, a); // expected-error {{'va_start' used in function with fixed args}}
}

void f2(int a, int b, ...)
{
    __builtin_va_list ap;

    __builtin_va_start(ap, 10); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
    __builtin_va_start(ap, a); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
    __builtin_va_start(ap, b);
}

void f3(float a, ...) { // expected-note 2{{parameter of type 'float' is declared here}}
    __builtin_va_list ap;

    __builtin_va_start(ap, a); // expected-warning {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
    __builtin_va_start(ap, (a)); // expected-warning {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
}


// stdarg: PR3075 and PR2531
void f4(const char *msg, ...) {
 __builtin_va_list ap;
 __builtin_stdarg_start((ap), (msg));
 __builtin_va_end (ap);
}

void f5() {
  __builtin_va_list ap;
  __builtin_va_start(ap,ap); // expected-error {{'va_start' used in function with fixed args}}
}

void f6(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap); // expected-error {{too few arguments to function}}
}

// PR3350
void
foo(__builtin_va_list authors, ...) {
  __builtin_va_start (authors, authors);
  (void)__builtin_va_arg(authors, int);
  __builtin_va_end (authors);
}

void f7(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  // FIXME: This error message is sub-par.
  __builtin_va_arg(ap, int) = 1; // expected-error {{expression is not assignable}}
  int *x = &__builtin_va_arg(ap, int); // expected-error {{cannot take the address of an rvalue}}
  __builtin_va_end(ap);
}

void f8(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  (void)__builtin_va_arg(ap, void); // expected-error {{second argument to 'va_arg' is of incomplete type 'void'}}
  __builtin_va_end(ap);
}

enum E { x = -1, y = 2, z = 10000 };
void f9(__builtin_va_list args)
{
    (void)__builtin_va_arg(args, float); // expected-warning {{second argument to 'va_arg' is of promotable type 'float'}}
    (void)__builtin_va_arg(args, enum E); // Don't warn here in C
    (void)__builtin_va_arg(args, short); // expected-warning {{second argument to 'va_arg' is of promotable type 'short'}}
    (void)__builtin_va_arg(args, char); // expected-warning {{second argument to 'va_arg' is of promotable type 'char'}}
}

void f10(int a, ...) {
  int i;
  __builtin_va_list ap;
  i = __builtin_va_start(ap, a); // expected-error {{assigning to 'int' from incompatible type 'void'}}
  __builtin_va_end(ap);
}

void f11(short s, ...) {  // expected-note {{parameter of type 'short' is declared here}}
  __builtin_va_list ap;
  __builtin_va_start(ap, s); // expected-warning {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
  __builtin_va_end(ap);
}

void f12(register int i, ...) {  // expected-note {{parameter of type 'int' is declared here}}
  __builtin_va_list ap;
  __builtin_va_start(ap, i); // expected-warning {{passing a parameter declared with the 'register' keyword to 'va_start' has undefined behavior}}
  __builtin_va_end(ap);
}

enum __attribute__((packed)) E1 {
  one1
};

void f13(enum E1 e, ...) {
  __builtin_va_list va;
  __builtin_va_start(va, e);
#ifndef MS
  // In Microsoft compatibility mode, all enum types are int, but in
  // non-ms-compatibility mode, this enumeration type will undergo default
  // argument promotions.
  // expected-note@-7 {{parameter of type 'enum E1' is declared here}}
  // expected-warning@-6 {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
#endif
  __builtin_va_end(va);
}
