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

// The pragma redeclares an existing declaration of the weak name, so it gets
// the same checks as a weak alias attribute on a redeclaration would.
void __defined_fn(void);
void defined_fn(void) {} // expected-note 2 {{previous definition is here}}
#pragma weak defined_fn = __defined_fn // expected-error {{redefinition of 'defined_fn'}} \
                                       // expected-warning {{attribute declaration must precede definition}}
void __defined_fn(void) {}

// The cloned variable takes the target's location, so diagnostics on it point
// at the target rather than at the pragma.
int __defined_var = 1; // expected-error {{alias definition of 'defined_var' after tentative definition}}
int defined_var; // expected-note {{previous definition is here}}
#pragma weak defined_var = __defined_var

int conflicting(long); // expected-note {{previous declaration is here}}
int __conflicting(int);
#pragma weak conflicting = __conflicting // expected-error {{conflicting types for 'conflicting'}}

static int internal(int);
int __internal(int);
#pragma weak internal = __internal // expected-error {{weak declaration cannot have internal linkage}}

extern _Thread_local int tls; // expected-note {{previous declaration is here}}
int __tls; // expected-error {{non-thread-local declaration of 'tls' follows thread-local declaration}}
#pragma weak tls = __tls

// GH35478, GH56760: a pre-existing declaration of the weak name must not make
// uses ambiguous.
int predecl(int), __predecl(int);
#pragma weak predecl = __predecl
int __predecl(int i) { return 0; }
int use_predecl(void) { return predecl(0); }
int predecl(int);
int use_predecl_redecl(void) { return predecl(0); }

// Repeating an alias with the same target is fine; a different target is not.
void dup_target1(void) {}
void dup_target2(void) {}
#pragma weak dup_alias = dup_target1 // expected-note {{previous definition is here}}
#pragma weak dup_alias = dup_target1
#pragma weak dup_alias = dup_target2 // expected-error {{redefinition of 'dup_alias'}}

#pragma weak fwd_dup_alias = fwd_dup_target1 // expected-note {{previous definition is here}}
#pragma weak fwd_dup_alias = fwd_dup_target2 // expected-error {{redefinition of 'fwd_dup_alias'}}
void fwd_dup_target1(void) {}
void fwd_dup_target2(void) {}
