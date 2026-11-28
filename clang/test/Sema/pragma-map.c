// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target

// Testing missing declarations.
#pragma map(d0, "D0") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(f9, "F9") // expected-warning{{failed to resolve '#pragma map' to a declaration}}

// Testing pragma map after decl.
static void sf0(void);
static int s0;
#pragma map(sf0, "SF0") // expected-warning{{'#pragma map' is applicable to symbols with external linkage only; not applied to 'sf0'}}
#pragma map(s0, "S0") // expected-warning{{'#pragma map' is applicable to symbols with external linkage only; not applied to 's0'}}

// Testing pragma map before decl.
#pragma map(sf1, "SF1") // expected-warning{{'#pragma map' is applicable to symbols with external linkage only; not applied to 'sf1'}}
#pragma map(s1, "S1") // expected-warning{{'#pragma map' is applicable to symbols with external linkage only; not applied to 's1'}}
static void sf1(void);
static int s1;

#pragma map(foo, "FOO1")
#pragma map(foo, "FOO2") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'foo'}}
int foo();

int bar();
#pragma map(bar, "BAR1")
#pragma map(bar, "BAR2") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'bar'}}

#pragma map(f, "F1") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'f'}}
int f() asm("F2");

int g() asm("G1");
#pragma map(g, "G2") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'g'}}

#pragma map(h, "H1") // expected-note {{previous declaration is here}}
int h();
int h() asm("H2"); // expected-error {{conflicting asm label}}

void func_defined() {
}
#pragma map(func_defined, "FUNC") // expected-warning {{'#pragma map' can only applied to unused symbols}}

void func_used() {
}
int var_used;

int func() {
  func_used();
  return var_used;
}
#pragma map(func_used, "FUNC1") // expected-warning {{'#pragma map' can only applied to unused symbols}}
#pragma map(var_used, "VAR1") // expected-warning {{'#pragma map' can only applied to unused symbols}}

static int var_static;
#pragma map(var_static,"VAR2") // expected-warning {{'#pragma map' is applicable to symbols with external linkage only; not applied to 'var_static'}}

// Testing __attribute__((overloadable)). Overloadable functions are mangled, so
// the pragma can only apply to the single overload that lacks the attribute.
// Cases that are accepted are covered in
// CodeGen/SystemZ/pragma-map-overloadable.c.

// All overloads are marked, so there is nothing to map.
void ov1(int) __attribute__((overloadable));
void ov1(double) __attribute__((overloadable));
#pragma map(ov1, "OV1") // expected-warning{{failed to resolve '#pragma map' to a declaration}}

#pragma map(ov2, "OV2") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
void ov2(int) __attribute__((overloadable));
void ov2(double) __attribute__((overloadable));

// Conflicting labels are diagnosed against the unmarked overload only.
void ov3(double) __attribute__((overloadable));
void ov3(int) asm("OV3A");
#pragma map(ov3, "OV3B") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'ov3'}}

// Using the unmarked overload prevents mapping.
void ov4(double) __attribute__((overloadable));
void ov4(int);
void use_ov4(void) {
  ov4(1);
}
#pragma map(ov4, "OV4") // expected-warning {{'#pragma map' can only applied to unused symbols}}
