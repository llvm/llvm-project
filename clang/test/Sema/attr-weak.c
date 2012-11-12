// RUN: %clang_cc1 -verify -fsyntax-only %s

extern int g0 __attribute__((weak));
extern int g1 __attribute__((weak_import));
int g2 __attribute__((weak));
int g3 __attribute__((weak_import)); // expected-warning {{'weak_import' attribute cannot be specified on a definition}}
int __attribute__((weak_import)) g4(void);
void __attribute__((weak_import)) g5(void) { 
}

struct __attribute__((weak)) s0 {}; // expected-warning {{'weak' attribute only applies to variables and functions}}
struct __attribute__((weak_import)) s1 {}; // expected-warning {{'weak_import' attribute only applies to variables and functions}}

static int x __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

// rdar://9538608
int C; // expected-note {{previous definition is here}}
extern int C __attribute__((weak_import)); // expected-warning {{an already-declared variable is made a weak_import declaration}}
