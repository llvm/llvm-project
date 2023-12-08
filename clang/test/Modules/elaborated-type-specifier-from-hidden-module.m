// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import ElaboratedTypeStructs.Empty; // The structs are now hidden.
struct S1 *x;
struct S2 *y;
struct S2 { int x; };
struct S3 *z;
// Incompatible definition.
struct S3 { float y; }; // expected-error {{has incompatible definitions}} // expected-note {{field has name}}
// expected-note@Inputs/elaborated-type-structs.h:3 {{field has name}}

@import ElaboratedTypeStructs.Structs;

void useS1(struct S1 *x);
void useS2(struct S2 *x);
void useS2(struct S2 *x);
