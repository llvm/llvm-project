// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.0-library -verify %s
// WaveActiveCountBits is unavailable before ShaderModel 6.0.

[shader("compute")]
[numthreads(8,8,1)]
unsigned foo() {
    // expected-error@#site {{'WaveActiveCountBits' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{'WaveActiveCountBits' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    return hlsl::WaveActiveCountBits(1); // #site
}
