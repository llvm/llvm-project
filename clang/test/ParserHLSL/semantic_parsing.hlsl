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

// '_' is a valid CPP identifier.
void MegaPony(int GI : _) { }

void GarguantuanPony(int GI : _1) { }

void CoolPony(int GI : A0A0) { }

void NicePony(int GI : A_0) { }

void CutePony(int GI : A00) { }

// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
void DoublePony(int GI : A00 B) { }

// Unicode can be used:
// https://timsong-cpp.github.io/cppwp/n3337/charname.allowed
void FrenchPony(int GI : garçon_de_café) { }
void UnicodePony(int GI : ℮) { }

// Since P1949 seems Emojis are not allowed, even if in the range
// mentioned in N3337.
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p1949r7.html

// expected-error@+2 {{unexpected character <U+1F60A>}}
// expected-error@+1 {{expected HLSL Semantic identifier}}
void UTFPony(int GI : 😊) { }

// expected-error@+1 {{character <U+1F60A> not allowed in an identifier}}
void SmilingPony(int GI : PonyWithA😊) { }
