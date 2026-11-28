// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target

// Cases that are accepted are covered in
// CodeGen/SystemZ/pragma-map-alias.c.

// An alias attribute that names its aliasee by source name marks the aliasee
// as used, so a later #pragma map on the aliasee is not applied.
void t1(void);
void a1(void) __attribute__((alias("t1")));
#pragma map(t1, "T1") // expected-warning {{'#pragma map' can only applied to unused symbols}}

// Same for variables.
extern int v1;
extern int va1 __attribute__((alias("v1")));
#pragma map(v1, "V1") // expected-warning {{'#pragma map' can only applied to unused symbols}}

// A function declared with the alias attribute is a definition, so the pragma
// is not applied to it when it appears after the alias declaration.
void t2(void) {}
void a2(void) __attribute__((alias("t2")));
#pragma map(a2, "A2") // expected-warning {{'#pragma map' can only applied to unused symbols}}

// Same for variables.
int v2 = 2;
extern int va2 __attribute__((alias("v2")));
#pragma map(va2, "VA2") // expected-warning {{'#pragma map' can only applied to unused symbols}}

// An alias that has already been used.
void t3(void) {}
void a3(void) __attribute__((alias("t3")));
void use_a3(void) { a3(); }
#pragma map(a3, "A3") // expected-warning {{'#pragma map' can only applied to unused symbols}}

// Conflicting asm label on the alias.
void t4(void) {}
#pragma map(a4, "A4") // expected-warning {{'#pragma map' has conflicting asm label; not applied to 'a4'}}
void a4(void) asm("A4X") __attribute__((alias("t4")));
