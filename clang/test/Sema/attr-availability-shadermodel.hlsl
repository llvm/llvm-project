// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-compute -fsyntax-only -verify %s

// Platform shader model, no environment parameter
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0(); // #fn6_0

__attribute__((availability(shadermodel, introduced = 5.1)))
unsigned fn5_1(); // #fn5_1

__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0();

// Platform shader model, environment parameter restricting earlier version,
// available in all environments in higher versions
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0_mix(); // #fn6_0_mix

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0_mix();

// Platform shader model, environment parameter restricting earlier version,
// never available in all environments in higher versions
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = mesh)))
unsigned fn6_0_stages1();  // #fn6_0_stages1

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn6_0_stages2(); // #fn6_0_stages2

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn5_0_stages();

// Strict parameter is not supported in HLSL
__attribute__((availability(shadermodel, strict, introduced = 6.0))) // expected-warning {{availability parameter 'strict' is not supported in HLSL}}
unsigned fn6_0_strict1();

__attribute__((availability(shadermodel, strict, introduced = 6.0, environment = pixel))) // expected-warning {{availability parameter 'strict' is not supported in HLSL}}
unsigned fn6_0_strict2();


[numthreads(4,1,1)]
int main() {
    // expected-error@#fn6_0_call {{'fn6_0' is only available on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0 {{'fn6_0' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    unsigned A1 = fn6_0(); // #fn6_0_call

    // expected-error@#fn5_1_call {{'fn5_1' is only available on Shader Model 5.1 or newer}}
    // expected-note@#fn5_1 {{'fn5_1' has been marked as being introduced in Shader Model 5.1 here, but the deployment target is Shader Model 5.0}}
    unsigned B1 = fn5_1(); // #fn5_1_call

    unsigned C1 = fn5_0();

    // expected-error@#fn6_0_mix_call {{'fn6_0_mix' is only available on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_mix {{'fn6_0_mix' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    unsigned A3 = fn6_0_mix(); // #fn6_0_mix_call

    unsigned B3 = fn5_0_mix();

    // expected-error@#fn6_0_stages1_call {{'fn6_0_stages1' is only available in compute shader environment on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_stages1 {{'fn6_0_stages1' has been marked as being introduced in Shader Model 6.0 in compute shader environment here, but the deployment target is Shader Model 5.0}}
    unsigned A5 = fn6_0_stages1(); // #fn6_0_stages1_call

    // expected-error@#fn6_0_stages2_call {{'fn6_0_stages2' is unavailable}}
    // expected-note@#fn6_0_stages2 {{'fn6_0_stages2' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 compute shader environment}}
    unsigned B5 = fn6_0_stages2(); // #fn6_0_stages2_call

    unsigned C5 = fn5_0_stages();

    unsigned A6 = fn6_0_strict1();

    unsigned B6 = fn6_0_strict1();

    return 0;
}
