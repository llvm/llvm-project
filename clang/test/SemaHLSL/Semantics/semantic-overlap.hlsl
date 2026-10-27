// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

struct Collide {
  float4 A : USER0;
// expected-note@-1 {{previous use is here}}
  float4 B : USER0;
// expected-error@-1 {{semantic index overlap USER0}}
// expected-note@-2 {{'B' used here}}
};

[shader("pixel")]
float4 collide(Collide C) : SV_Target { return C.A; }
// expected-note@-1 {{'C' declared here}}

// Semantic names are case-insensitive.
struct CollideCase {
  float4 A : user0;
// expected-note@-1 {{previous use is here}}
  float4 B : USER0;
// expected-error@-1 {{semantic index overlap USER0}}
// expected-note@-2 {{'B' used here}}
};

[shader("pixel")]
float4 collide_case(CollideCase C) : SV_Target { return C.A; }
// expected-note@-1 {{'C' declared here}}

struct CollidePosition {
  float4 A : SV_Position;
// expected-note@-1 {{previous use is here}}
  float4 B : sv_position;
// expected-error@-1 {{semantic index overlap sv_position0}}
// expected-note@-2 {{'B' used here}}
};

[shader("pixel")]
float4 collide_position(CollidePosition C) : SV_Target { return C.A; }
// expected-note@-1 {{'C' declared here}}

// The array reserves USER0 and USER1.
struct CollideArray {
  float4 A[2] : USER0;
// expected-note@-1 {{previous use is here}}
  float4 B : USER1;
// expected-error@-1 {{semantic index overlap USER1}}
// expected-note@-2 {{'B' used here}}
};

[shader("pixel")]
float4 collide_array(CollideArray C) : SV_Target { return C.B; }
// expected-note@-1 {{'C' declared here}}

// All four elements of the nested array reserve semantic indices.
struct CollideNestedArray {
  float4 A[2][2] : USER0;
// expected-note@-1 {{previous use is here}}
  float4 B : USER3;
// expected-error@-1 {{semantic index overlap USER3}}
// expected-note@-2 {{'B' used here}}
};

[shader("pixel")]
float4 collide_nested_array(CollideNestedArray C) : SV_Target { return C.B; }
// expected-note@-1 {{'C' declared here}}

struct NoCollideNestedArray {
  float4 A[2][2] : USER0;
  float4 B : USER4;
};

[shader("pixel")]
float4 no_collide_nested_array(NoCollideNestedArray C) : SV_Target {
  return C.B;
}

struct NoCollide {
  float4 A : USER0;
  float4 B : USER1;
};

[shader("pixel")]
float4 no_collide(NoCollide C) : SV_Target { return C.A; }

// The same semantic index is allowed in separate input and output signatures.
[shader("vertex")]
float4 separate_signatures(float4 C : USER0) : USER0 { return C; }
