// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DMISMATCHED -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DMISSING -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DAMBIGUOUS -verify

#ifdef MISSING
// expected-error@*:* {{missing entry point definition 'foo'}}
[numthreads(1,1,1), shader("compute")]
void oof() {

}
#else
#ifdef MISMATCHED
// expected-error@+1 {{'shader' attribute on entry function does not match the target profile}}
[numthreads(1,1,1), shader("mesh")]
#else
[numthreads(1,1,1), shader("compute")]
#endif
void foo() {

}

#ifdef AMBIGUOUS
// expected-error@+2 {{ambiguous entry point definition 'foo'}}
[numthreads(1,1,1), shader("compute")]
void foo(unsigned int GI : SV_GroupIndex) {

}
// expected-note@*:* {{previous 'foo' definition is here}}
#endif
#endif

