// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-apple-darwin9

void f1(void) {
  const __builtin_va_list args2;
  (void)__builtin_va_arg(args2, int); // expected-error {{first argument to 'va_arg' is of type 'const __builtin_va_list' and not 'va_list'}}
}

void f2(int a, ...) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a); // expected-error {{'__builtin_ms_va_start' used in System V ABI function}}
}

void __attribute__((ms_abi)) g1(int a) {
  __builtin_ms_va_list ap;

  __builtin_ms_va_start(ap, a, a); // expected-error {{too many arguments to function}}
  __builtin_ms_va_start(ap, a); // expected-error {{'va_start' used in function with fixed args}}
}

void __attribute__((ms_abi)) g2(int a, int b, ...) {
  __builtin_ms_va_list ap;

  __builtin_ms_va_start(ap, 10); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
  __builtin_ms_va_start(ap, a); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
  __builtin_ms_va_start(ap, b);
}

void __attribute__((ms_abi)) g3(float a, ...) { // expected-note 2{{parameter of type 'float' is declared here}}
  __builtin_ms_va_list ap;

  __builtin_ms_va_start(ap, a); // expected-warning {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
  __builtin_ms_va_start(ap, (a)); // expected-warning {{passing an object that undergoes default argument promotion to 'va_start' has undefined behavior}}
}

void __attribute__((ms_abi)) g5(void) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, ap); // expected-error {{'va_start' used in function with fixed args}}
}

void __attribute__((ms_abi)) g6(int a, ...) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap); // expected-error {{too few arguments to function}}
}

void __attribute__((ms_abi))
bar(__builtin_ms_va_list authors, ...) {
  __builtin_ms_va_start(authors, authors);
  (void)__builtin_va_arg(authors, int);
  __builtin_ms_va_end(authors);
}

void __attribute__((ms_abi)) g7(int a, ...) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // FIXME: This error message is sub-par.
  __builtin_va_arg(ap, int) = 1; // expected-error {{expression is not assignable}}
  int *x = &__builtin_va_arg(ap, int); // expected-error {{cannot take the address of an rvalue}}
  __builtin_ms_va_end(ap);
}

void __attribute__((ms_abi)) g8(int a, ...) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  (void)__builtin_va_arg(ap, void); // expected-error {{second argument to 'va_arg' is of incomplete type 'void'}}
  __builtin_ms_va_end(ap);
}

enum E { x = -1, y = 2, z = 10000 };
void __attribute__((ms_abi)) g9(__builtin_ms_va_list args) {
  (void)__builtin_va_arg(args, float); // expected-warning {{second argument to 'va_arg' is of promotable type 'float'}}
  (void)__builtin_va_arg(args, enum E); // no-warning
  (void)__builtin_va_arg(args, short); // expected-warning {{second argument to 'va_arg' is of promotable type 'short'}}
  (void)__builtin_va_arg(args, char); // expected-warning {{second argument to 'va_arg' is of promotable type 'char'}}
}

void __attribute__((ms_abi)) g10(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, a); // expected-error {{'va_start' used in Win64 ABI function}}
}
