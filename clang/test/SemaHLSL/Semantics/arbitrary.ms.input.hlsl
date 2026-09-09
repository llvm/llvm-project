// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-mesh -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-mesh -finclude-default-header -x hlsl -verify -o - %s

[numthreads(1,1,1)]
void main(uint a : A) {
// expected-error@-1 {{semantic 'A' is not supported in mesh shader inputs; it is only available as an output, a patch constant or a primitive}}
}
