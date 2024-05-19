// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.0-library -verify %s
// WaveActiveCountBits is unavailable before ShaderModel 6.0.

<<<<<<< HEAD
[shader("compute")]
[numthreads(8,8,1)]
unsigned foo() {
    // expected-error@#site {{'WaveActiveCountBits' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{'WaveActiveCountBits' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    return hlsl::WaveActiveCountBits(1); // #site
=======
unsigned foo(bool b) {
    // expected-warning@#site {{'WaveActiveCountBits' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{'WaveActiveCountBits' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    // expected-note@#site {{enclose 'WaveActiveCountBits' in a __builtin_available check to silence this warning}}
    return hlsl::WaveActiveCountBits(b); // #site
>>>>>>> 3f33c4c14e79e68007cf1460e4a0e606eb199da5
}
