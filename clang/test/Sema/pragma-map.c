// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target

// Testing missing declarations.
#pragma map(d0, "D0") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(f9, "F9") // expected-warning{{failed to resolve '#pragma map' to a declaration}}

// Testing pragma map after decl.
static void sf0(void);
static int s0;
#pragma map(sf0, "SF0") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 'sf0'}}
#pragma map(s0, "S0") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 's0'}}

// Testing pragma map before decl.
#pragma map(sf1, "SF1") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 'sf1'}}
#pragma map(s1, "S1") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 's1'}}
static void sf1(void);
static int s1;

#pragma map(foo, "FOO1")
#pragma map(foo, "FOO2") // expected-warning {{#pragma map has conflicting asm label; not applied to 'foo'}}
int foo();

int bar();
#pragma map(bar, "BAR1")
#pragma map(bar, "BAR2") // expected-warning {{#pragma map has conflicting asm label; not applied to 'bar'}}

#pragma map(f, "F1") // expected-warning {{#pragma map has conflicting asm label; not applied to 'f'}}
int f() asm("F2");

int g() asm("G1");
#pragma map(g, "G2") // expected-warning {{#pragma map has conflicting asm label; not applied to 'g'}}

#pragma map(h, "H1") // expected-note {{previous declaration is here}}
int h();
int h() asm("H2"); // expected-error {{conflicting asm label}}

void func_defined() {
}
#pragma map(func_defined, "FUNC") // expected-warning {{#pragma map can only applied to unused symbols}}

void func_used() {
}
int var_used;

int func() {
  func_used();
  return var_used;
}
#pragma map(func_used, "FUNC1") // expected-warning {{#pragma map can only applied to unused symbols}}
#pragma map(var_used, "VAR1") // expected-warning {{#pragma map can only applied to unused symbols}}

static int var_static;
#pragma map(var_static,"VAR2") // expected-warning {{#pragma map is applicable to symbols with external linkage only; not applied to 'var_static'}}


