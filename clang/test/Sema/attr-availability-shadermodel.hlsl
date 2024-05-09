// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-compute -fsyntax-only -verify %s

// Platform shader model, no environment parameter
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0(); // #fn6_0

__attribute__((availability(shadermodel, introduced = 5.1)))
unsigned fn5_1(); // #fn5_1

__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0();

// Platform shader model, no environment parameter
// - with strict flag (executes different code path)
__attribute__((availability(shadermodel, strict, introduced = 6.0)))
unsigned fn6_0_s(); // #fn6_0_s

__attribute__((availability(shadermodel, strict, introduced = 5.1)))
unsigned fn5_1_s(); // #fn5_1_s

__attribute__((availability(shadermodel, strict, introduced = 5.0)))
unsigned fn5_0_s();

// Platform shader model, environment parameter restricting earlier version,
// available in all environments in higher versions
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0_mix(); // #fn6_0_mix

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0_mix();

// Platform shader model, environment parameter restricting earlier version,
// available in all environments in higher versions
// - with strict flag (executes different code path)
__attribute__((availability(shadermodel, strict, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, strict, introduced = 6.0)))
unsigned fn6_0_mix_s(); // #fn6_0_mix_s

__attribute__((availability(shadermodel, strict, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, strict, introduced = 5.0)))
unsigned fn5_0_mix_s();

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

// Platform shader model, environment parameter restricting earlier version,
// never available in all environments in higher versions
// - with strict flag (executes different code path)
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = mesh)))
unsigned fn6_0_stages1_s();  // #fn6_0_stages1_s

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn6_0_stages2_s(); // #fn6_0_stages2_s

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn5_0_stages_s();

[numthreads(4,1,1)]
int main() {
    // expected-warning@#fn6_0_call {{'fn6_0' is only available on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0 {{'fn6_0' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_call {{enclose 'fn6_0' in a __builtin_available check to silence this warning}}
    unsigned A1 = fn6_0(); // #fn6_0_call

    // expected-warning@#fn5_1_call {{'fn5_1' is only available on Shader Model 5.1 or newer}}
    // expected-note@#fn5_1 {{'fn5_1' has been marked as being introduced in Shader Model 5.1 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn5_1_call {{enclose 'fn5_1' in a __builtin_available check to silence this warning}}
    unsigned B1 = fn5_1(); // #fn5_1_call

    unsigned C1 = fn5_0();

    // expected-error@#fn6_0_s_call {{'fn6_0_s' is unavailable: introduced in Shader Model 6.0 compute shader}}
    // expected-note@#fn6_0_s {{'fn6_0_s' has been explicitly marked unavailable here}}
    unsigned A2 = fn6_0_s(); // #fn6_0_s_call

    // expected-error@#fn5_1_s_call {{'fn5_1_s' is unavailable: introduced in Shader Model 5.1 compute shader}}
    // expected-note@#fn5_1_s {{'fn5_1_s' has been explicitly marked unavailable here}}
    unsigned B2 = fn5_1_s(); // #fn5_1_s_call

    unsigned C2 = fn5_0_s();

    // expected-warning@#fn6_0_mix_call {{'fn6_0_mix' is only available on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_mix {{'fn6_0_mix' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_mix_call {{enclose 'fn6_0_mix' in a __builtin_available check to silence this warning}}
    unsigned A3 = fn6_0_mix(); // #fn6_0_mix_call

    unsigned B3 = fn5_0_mix();

    // expected-error@#fn6_0_mix_s_call {{'fn6_0_mix_s' is unavailable: introduced in Shader Model 6.0 compute shader}}
    // expected-note@#fn6_0_mix_s {{'fn6_0_mix_s' has been explicitly marked unavailable here}}
    unsigned A4 = fn6_0_mix_s(); // #fn6_0_mix_s_call

    unsigned B4 = fn5_0_mix_s();

    // expected-warning@#fn6_0_stages1_call {{'fn6_0_stages1' is only available in compute shader environment on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_stages1 {{'fn6_0_stages1' has been marked as being introduced in Shader Model 6.0 in compute shader environment here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_stages1_call {{enclose 'fn6_0_stages1' in a __builtin_available check to silence this warning}}
    unsigned A5 = fn6_0_stages1(); // #fn6_0_stages1_call

    // expected-warning@#fn6_0_stages2_call {{'fn6_0_stages2' is unavailable}}
    // expected-note@#fn6_0_stages2 {{'fn6_0_stages2' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 compute shader environment}}
    // expected-note@#fn6_0_stages2_call {{enclose 'fn6_0_stages2' in a __builtin_available check to silence this warning}}
    unsigned B5 = fn6_0_stages2(); // #fn6_0_stages2_call

    unsigned C5 = fn5_0_stages();

    // expected-warning@#fn6_0_stages1_s_call {{'fn6_0_stages1_s' is only available in compute shader environment on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_stages1_s {{'fn6_0_stages1_s' has been marked as being introduced in Shader Model 6.0 in compute shader environment here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_stages1_s_call {{enclose 'fn6_0_stages1_s' in a __builtin_available check to silence this warning}}
    unsigned A6 = fn6_0_stages1_s(); // #fn6_0_stages1_s_call

    // expected-warning@#fn6_0_stages2_s_call {{'fn6_0_stages2_s' is unavailable}}
    // expected-note@#fn6_0_stages2_s {{'fn6_0_stages2_s' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 compute shader environment}}
    // expected-note@#fn6_0_stages2_s_call {{enclose 'fn6_0_stages2_s' in a __builtin_available check to silence this warning}}
    unsigned B6 = fn6_0_stages2_s(); // #fn6_0_stages2_s_call

    unsigned C6 = fn5_0_stages_s();

    return 0;
}
