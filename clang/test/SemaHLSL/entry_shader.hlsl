// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DSHADER='"anyHit"' -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DSHADER='"compute"'

// expected-error@+1 {{'shader' attribute on entry function does not match the pipeline stage}}
[numthreads(1,1,1), shader(SHADER)]
void foo() {

}
