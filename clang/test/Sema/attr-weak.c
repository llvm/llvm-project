// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -fsyntax-only %s

extern int f0(void) __attribute__((weak));
extern int g0 __attribute__((weak));
extern int g1 __attribute__((weak_import));
int f2(void) __attribute__((weak));
int g2 __attribute__((weak));
int g3 __attribute__((weak_import)); // expected-warning {{'weak_import' attribute cannot be specified on a definition}}
int __attribute__((weak_import)) g4(void);
void __attribute__((weak_import)) g5(void) {
}

struct __attribute__((weak)) s0 {}; // expected-warning {{'weak' attribute only applies to variables, functions, and classes}}
struct __attribute__((weak_import)) s1 {}; // expected-warning {{'weak_import' attribute only applies to variables and functions}}

static int f(void) __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}
static int x __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

int C; // expected-note {{previous definition is here}}
extern int C __attribute__((weak_import)); // expected-warning {{'C' cannot be declared 'weak_import'}}

int C2; // expected-note {{previous definition is here}}
extern int C2;
extern int C2 __attribute__((weak_import)); // expected-warning {{'C2' cannot be declared 'weak_import'}}

static int pr14946_x;
extern int pr14946_x  __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

static void pr14946_f(void);
void pr14946_f(void) __attribute__((weak)); // expected-error {{weak declaration cannot have internal linkage}}

// An ifunc declaration is valid on its own, but cannot also be weak.
void *resolver(void) { return 0; }
void ifunc_function(void) __attribute__((ifunc("resolver")));
void ifunc_weak(void) __attribute__((ifunc("resolver"), weak));
// expected-error@-1 {{'weak' and 'ifunc' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}

// A weak attribute inherited from an earlier declaration also conflicts.
void inherited_weak(void) __attribute__((weak)); // expected-note {{conflicting attribute is here}}
void inherited_weak(void) __attribute__((ifunc("resolver")));
// expected-error@-1 {{'ifunc' and 'weak' attributes are not compatible}}

// Adding weak after an ifunc definition is already diagnosed and ignored.
void late_weak(void) __attribute__((ifunc("resolver"))); // expected-note {{previous definition is here}}
void late_weak(void) __attribute__((weak)); // expected-warning {{attribute declaration must precede definition}}
