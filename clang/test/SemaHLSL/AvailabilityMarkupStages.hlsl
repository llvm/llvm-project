// RUN: %clang_cc1 -triple dxil-unknown-shadermodel5.0-compute -verify %s

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned fn6_0_all(); // #fn6_0_all_def

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned fn5_0_all();

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = mesh)))
unsigned fn6_0_stages1();  // #fn6_0_stages1_def

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn6_0_stages2(); // #fn6_0_stages2_def

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned fn5_0_stages();

void fn() {

    // expected-warning@#fn6_0_all_call {{'fn6_0_all' is only available on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_all_def {{'fn6_0_all' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_all_call {{enclose 'fn6_0_all' in a __builtin_available check to silence this warning}}
    unsigned A = fn6_0_all(); // #fn6_0_all_call

    unsigned B = fn5_0_all();

    // expected-warning@#fn6_0_stages1_call {{'fn6_0_stages1' is only available in compute shader environment on Shader Model 6.0 or newer}}
    // expected-note@#fn6_0_stages1_def {{'fn6_0_stages1' has been marked as being introduced in Shader Model 6.0 in compute shader environment here, but the deployment target is Shader Model 5.0}}
    // expected-note@#fn6_0_stages1_call {{enclose 'fn6_0_stages1' in a __builtin_available check to silence this warning}}
    unsigned C = fn6_0_stages1(); // #fn6_0_stages1_call

    // expected-warning@#fn6_0_stages2_call {{'fn6_0_stages2' is unavailable}}
    // expected-note@#fn6_0_stages2_def {{'fn6_0_stages2' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 compute shader environment}}
    // expected-note@#fn6_0_stages2_call {{enclose 'fn6_0_stages2' in a __builtin_available check to silence this warning}}
    unsigned E = fn6_0_stages2(); // #fn6_0_stages2_call

    unsigned D = fn5_0_stages();
}
