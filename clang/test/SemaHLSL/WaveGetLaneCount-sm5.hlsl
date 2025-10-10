// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.1-library -verify %s

[shader("compute")]
[numthreads(8,8,1)]
unsigned foo() {
    // expected-error@#site {{'WaveGetLaneCount' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{'WaveGetLaneCount' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.1}}
    return hlsl::WaveGetLaneCount(); // #site
}
