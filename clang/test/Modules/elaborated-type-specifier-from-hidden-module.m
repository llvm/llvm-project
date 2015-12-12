// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import ElaboratedTypeStructs.Empty; // The structs are now hidden.
struct S1 *x;
struct S2 *y;
// FIXME: compatible definition should not be an error.
struct S2 { int x; }; // expected-error {{redefinition}}
struct S3 *z;
// Incompatible definition.
struct S3 { float y; }; // expected-error {{redefinition}}
// expected-note@elaborated-type-structs.h:* 2 {{previous definition is here}}

@import ElaboratedTypeStructs.Structs;

void useS1(struct S1 *x);
void useS2(struct S2 *x);
void useS2(struct S2 *x);
