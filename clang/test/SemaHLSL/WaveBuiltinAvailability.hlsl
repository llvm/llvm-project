// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.0-library -verify %s
// Wave intrinsics are unavailable before ShaderModel 6.0.

[shader("compute")]
[numthreads(8,8,1)]
void foo() {
    // expected-error@#WaveActiveCountBits {{'WaveActiveCountBits' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveActiveCountBits' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    unsigned tmp = hlsl::WaveActiveCountBits(1); // #WaveActiveCountBits

    // expected-error@#WaveActiveMax {{'WaveActiveMax' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveActiveMax' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float a = hlsl::WaveActiveMax(1.0f); // #WaveActiveMax

    // expected-error@#WaveActiveMin {{'WaveActiveMin' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveActiveMin' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float b = hlsl::WaveActiveMin(1.0f); // #WaveActiveMin

    // expected-error@#WaveActiveProduct {{'WaveActiveProduct' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveActiveProduct' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float c = hlsl::WaveActiveProduct(1.0f); // #WaveActiveProduct

    // expected-error@#WaveActiveSum {{'WaveActiveSum' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveActiveSum' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float d = hlsl::WaveActiveSum(1.0f); // #WaveActiveSum

    // expected-error@#WavePrefixProduct {{'WavePrefixProduct' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WavePrefixProduct' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float e = hlsl::WavePrefixProduct(1.0f); // #WavePrefixProduct

    // expected-error@#WavePrefixSum {{'WavePrefixSum' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WavePrefixSum' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float f = hlsl::WavePrefixSum(1.0f); // #WavePrefixSum

    // expected-error@#WaveReadLaneAt {{'WaveReadLaneAt' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'WaveReadLaneAt' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float g = hlsl::WaveReadLaneAt(1.0f, 0u); // #WaveReadLaneAt
}
