// RUN: %clang_cc1 -fsyntax-only %s -Wincompatible-function-pointer-types-strict -verify=soft,strict
// RUN: %clang_cc1 -fsyntax-only %s -Werror=incompatible-function-pointer-types-strict -verify=hard,strict
// RUN: %clang_cc1 -fsyntax-only %s -Wincompatible-function-pointer-types -verify=nonstrict
// nonstrict-no-diagnostics

enum E { A = -1, B };
typedef enum E (*fn_a_t)(void);
typedef void (*fn_b_t)(void);

int a(void) { return 0; }
void __attribute__((noreturn)) b(void) { while (1); }

void fa(fn_a_t x) {} // strict-note {{passing argument to parameter 'x' here}}
void fb(fn_b_t x) {}

void baz(void) {
  fa(&a); // soft-warning {{incompatible function pointer types passing 'int (*)(void)' to parameter of type 'fn_a_t' (aka 'enum E (*)(void)')}} \
             hard-error {{incompatible function pointer types passing 'int (*)(void)' to parameter of type 'fn_a_t' (aka 'enum E (*)(void)')}}
  fb(&b); // no-warning
}
