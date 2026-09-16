// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -fsyntax-only -verify %s

[shader("compute")][numthreads(1,1,1)]
void empty_input(uint V[0] : SV_GroupIndex) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' cannot be applied to a zero-sized array}}
// expected-note@-2 {{'V' declared here}}

[shader("pixel")]
void empty_output(out float4 V[0] : SV_Target) {}
// expected-error@-1 {{semantic 'SV_Target' cannot be applied to a zero-sized array}}
// expected-note@-2 {{'V' declared here}}

// A zero in any array dimension makes the semantic range empty.
[shader("compute")][numthreads(1,1,1)]
void empty_inner_dimension(uint V[2][0] : SV_GroupIndex) {}
// expected-error@-1 {{semantic 'SV_GroupIndex' cannot be applied to a zero-sized array}}
// expected-note@-2 {{'V' declared here}}

// User semantics and typedefs must not bypass the check.
typedef float4 EmptyArray[0];

[shader("vertex")]
float4 empty_user_input(EmptyArray V : USER) : SV_Position { return 0; }
// expected-error@-1 {{semantic 'USER' cannot be applied to a zero-sized array}}
// expected-note@-2 {{'V' declared here}}

struct EmptyOutput {
  EmptyArray V : SV_Target;
// expected-error@-1 {{semantic 'SV_Target' cannot be applied to a zero-sized array}}
// expected-note@-2 {{'V' used here}}
};

[shader("pixel")]
EmptyOutput empty_return() { return (EmptyOutput)0; }
