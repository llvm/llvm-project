// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.0-library -verify %s
// WaveActiveCountBits is unavailable before ShaderModel 6.0.

unsigned foo(bool b) {
    // expected-warning@#site {{'WaveActiveCountBits' is only available on HLSL ShaderModel 6.0 or newer}}
    // expected-note@hlsl/hlsl_intrinsics.h:* {{'WaveActiveCountBits' has been marked as being introduced in HLSL ShaderModel 6.0 here, but the deployment target is HLSL ShaderModel 5.0}}
    // expected-note@#site {{enclose 'WaveActiveCountBits' in a __builtin_available check to silence this warning}}
    return hlsl::WaveActiveCountBits(b); // #site
}
