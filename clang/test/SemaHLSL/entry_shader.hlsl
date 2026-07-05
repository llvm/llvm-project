// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DSHADER='"mesh"' -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s -DSHADER='"compute"'

// expected-error@+1 {{'shader' attribute on entry function does not match the target profile}}
[numthreads(1,1,1), shader(SHADER)]
void foo() {

}
