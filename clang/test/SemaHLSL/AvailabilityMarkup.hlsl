// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-library -verify %s

__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0(); // #fn6_0

__attribute__((availability(shadermodel, introduced = 5.1)))
unsigned fn5_1(); // #fn5_1

__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0();

void fn() {
    // expected-warning@#fn6_0_site {{'fn6_0' is only available on HLSL ShaderModel 6.0 or newer}}
    // expected-note@#fn6_0 {{'fn6_0' has been marked as being introduced in HLSL ShaderModel 6.0 here, but the deployment target is HLSL ShaderModel 5.0}}
    // expected-note@#fn6_0_site {{enclose 'fn6_0' in a __builtin_available check to silence this warning}}
    unsigned A = fn6_0(); // #fn6_0_site

    // expected-warning@#fn5_1_site {{'fn5_1' is only available on HLSL ShaderModel 5.1 or newer}}
    // expected-note@#fn5_1 {{'fn5_1' has been marked as being introduced in HLSL ShaderModel 5.1 here, but the deployment target is HLSL ShaderModel 5.0}}
    // expected-note@#fn5_1_site {{enclose 'fn5_1' in a __builtin_available check to silence this warning}}
    unsigned B = fn5_1(); // #fn5_1_site

    unsigned C = fn5_0();
}

