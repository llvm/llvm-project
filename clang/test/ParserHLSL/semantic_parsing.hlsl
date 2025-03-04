// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -verify

// expected-error@+1 {{expected HLSL Semantic identifier}}
void Entry(int GI : ) { }

// expected-error@+1 {{unknown HLSL semantic 'sv_iwantapony'}}
void Pony(int GI : SV_IWantAPony) { }
