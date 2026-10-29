// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

void __both3(void);
#pragma weak both3 = __both3 // expected-note {{previous definition}}
void both3(void) __attribute((alias("__both3"))); // expected-error {{redefinition of 'both3'}}
void __both3(void) {}

void __a3(void) __attribute((noinline));
#pragma weak a3 = __a3 // expected-note {{previous definition}}
void a3(void) __attribute((alias("__a3"))); // expected-error {{redefinition of 'a3'}}
void __a3(void) {}

void *resolver(void) { return 0; }

// Pragma weak can be applied before or after the ifunc declaration.
#pragma weak pragma_before // expected-note {{conflicting attribute is here}}
void pragma_before(void) __attribute__((ifunc("resolver")));
// expected-error@-1 {{'ifunc' and 'weak' attributes are not compatible}}

void pragma_after(void) __attribute__((ifunc("resolver")));
// expected-error@-1 {{'ifunc' and 'weak' attributes are not compatible}}
#pragma weak pragma_after // expected-note {{conflicting attribute is here}}

// The ifunc attribute is not inherited by later declarations.
void pragma_after_redecl(void) __attribute__((ifunc("resolver")));
// expected-error@-1 {{'ifunc' and 'weak' attributes are not compatible}}
void pragma_after_redecl(void);
#pragma weak pragma_after_redecl // expected-note {{conflicting attribute is here}}

// The ifunc attribute can also be introduced on a redeclaration.
void pragma_ifunc_on_redecl(void);
void pragma_ifunc_on_redecl(void) __attribute__((ifunc("resolver")));
// expected-error@-1 {{'ifunc' and 'weak' attributes are not compatible}}
#pragma weak pragma_ifunc_on_redecl // expected-note {{conflicting attribute is here}}

// A weak alias of an ifunc does not make the ifunc itself weak.
#pragma weak pragma_alias = ifunc_alias_target
void ifunc_alias_target(void) __attribute__((ifunc("resolver")));

// The weak name is already defined: a definition cannot also be an alias.
void __defined_fn(void);
void defined_fn(void) {}
#pragma weak defined_fn = __defined_fn // expected-error {{definition 'defined_fn' cannot also be an alias}}
void __defined_fn(void) {}

int __defined_var = 1;
int defined_var;
#pragma weak defined_var = __defined_var // expected-error {{definition 'defined_var' cannot also be an alias}}

// GH35478, GH56760: a pre-existing declaration of the weak name must not make
// uses ambiguous.
int predecl(int), __predecl(int);
#pragma weak predecl = __predecl
int __predecl(int i) { return 0; }
int use_predecl(void) { return predecl(0); }
