// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-mesh -fsyntax-only -verify %s

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
unsigned f5(); // #f5

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
unsigned f8(); // #f8

[numthreads(4,1,1)]
int main() {
    // expected-warning@#f1_call {{'f1' is only available on Shader Model 6.0 or newer}}
    // expected-note@#f1 {{'f1' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#f1_call {{enclose 'f1' in a __builtin_available check to silence this warning}}
    unsigned A = f1(); // #f1_call

    // expected-warning@#f2_call {{'f2' is only available on Shader Model 5.1 or newer}}
    // expected-note@#f2 {{'f2' has been marked as being introduced in Shader Model 5.1 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#f2_call {{enclose 'f2' in a __builtin_available check to silence this warning}}
    unsigned B = f2(); // #f2_call

    unsigned C = f3();

    // expected-warning@#f4_call {{'f4' is only available on Shader Model 6.0 or newer}}
    // expected-note@#f4 {{'f4' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#f4_call {{enclose 'f4' in a __builtin_available check to silence this warning}}
    unsigned D = f4(); // #f4_call

    unsigned E = f5(); // #f5_call

    unsigned F = f6(); // #f6_call

    // expected-warning@#f7_call {{'f7' is only available in mesh shader environment on Shader Model 6.0 or newer}}
    // expected-note@#f7 {{'f7' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 mesh shader environment}}
    // expected-note@#f7_call {{enclose 'f7' in a __builtin_available check to silence this warning}}
    unsigned G = f7(); // #f7_call

    // expected-warning@#f8_call {{'f8' is only available in mesh shader environment on Shader Model 6.0 or newer}}
    // expected-note@#f8 {{'f8' has been marked as being introduced in Shader Model 6.0 in mesh shader environment here, but the deployment target is Shader Model 5.0 mesh shader environment}}
    // expected-note@#f8_call {{enclose 'f8' in a __builtin_available check to silence this warning}}
    unsigned H = f8(); // #f8_call

    return 0;
}
