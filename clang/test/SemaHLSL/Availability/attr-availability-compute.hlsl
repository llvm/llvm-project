// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-compute -fsyntax-only -verify %s

// Platform shader model, no environment parameter
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned f1(); // #f1

__attribute__((availability(shadermodel, introduced = 5.1)))
unsigned f2(); // #f2

__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned f3();

// Platform shader model, environment parameter restricting earlier version,
// available in all environments in higher versions
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0)))
unsigned f4(); // #f4

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0)))
unsigned f5();

// Platform shader model, environment parameter restricting earlier version,
// never available in all environments in higher versions
__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = mesh)))
unsigned f6();  // #f6

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned f7(); // #f7

__attribute__((availability(shadermodel, introduced = 2.0, environment = pixel)))
__attribute__((availability(shadermodel, introduced = 5.0, environment = compute)))
__attribute__((availability(shadermodel, introduced = 6.0, environment = mesh)))
unsigned f8();

[numthreads(4,1,1)]
int main() {
    // expected-error@#f1_call {{'f1' is only available on Shader Model 6.0 or newer}}
    // expected-note@#f1 {{'f1' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    unsigned A = f1(); // #f1_call

    // expected-error@#f2_call {{'f2' is only available on Shader Model 5.1 or newer}}
    // expected-note@#f2 {{'f2' has been marked as being introduced in Shader Model 5.1 here, but the deployment target is Shader Model 5.0}}
    unsigned B = f2(); // #f2_call

    unsigned C = f3();

    // expected-error@#f4_call {{'f4' is only available on Shader Model 6.0 or newer}}
    // expected-note@#f4 {{'f4' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    unsigned D = f4(); // #f4_call

    unsigned E = f5();

    // expected-error@#f6_call {{'f6' is only available in compute environment on Shader Model 6.0 or newer}}
    // expected-note@#f6 {{'f6' has been marked as being introduced in Shader Model 6.0 in compute environment here, but the deployment target is Shader Model 5.0}}
    unsigned F = f6(); // #f6_call

    // expected-error@#f7_call {{'f7' is unavailable}}
    // expected-note@#f7 {{'f7' has been marked as being introduced in Shader Model 6.0 in mesh environment here, but the deployment target is Shader Model 5.0 compute environment}}
    unsigned G = f7(); // #f7_call

    unsigned H = f8();

    return 0;
}
