// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl -o - %s -verify

// expected-error@+1 {{expected HLSL Semantic identifier}}
void Entry(int GI : ) { }

// expected-error@+1 {{unknown HLSL semantic 'SV_IWantAPony'}}
void Pony(int GI : SV_IWantAPony) { }

// expected-error@+3 {{expected HLSL Semantic identifier}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
void SuperPony(int GI : 0) { }

// expected-error@+1 {{unknown HLSL semantic '_'}}
void MegaPony(int GI : _) { }

// expected-error@+1 {{unknown HLSL semantic 'A0A'}}
void CoolPony(int GI : A0A0) { }

// expected-error@+1 {{unknown HLSL semantic 'A_'}}
void NicePony(int GI : A_0) { }

// expected-error@+1 {{unknown HLSL semantic 'A'}}
void CutePony(int GI : A00) { }

// expected-error@+3 {{unknown HLSL semantic 'A'}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
void DoublePony(int GI : A00 B) { }

// expected-error@+1 {{unknown HLSL semantic 'é'}}
void BigPony(int GI : é) { }

// expected-error@+2 {{unexpected character <U+1F60A>}}
// expected-error@+1 {{expected HLSL Semantic identifier}}
void UTFPony(int GI : 😊) { }

// expected-error@+2 {{character <U+1F60A> not allowed in an identifier}}
// expected-error@+1 {{unknown HLSL semantic 'PonyWithA😊'}}
void SmilingPony(int GI : PonyWithA😊) { }
