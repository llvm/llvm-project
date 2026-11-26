// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel5.0-library -verify %s
// Quad intrinsics are unavailable before ShaderModel 6.0.

[shader("compute")]
[numthreads(2,2,1)]
void foo() {
    // expected-error@#QuadReadAcrossX {{'QuadReadAcrossX' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'QuadReadAcrossX' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float x = hlsl::QuadReadAcrossX(1.0f); // #QuadReadAcrossX

    // expected-error@#QuadReadAcrossY {{'QuadReadAcrossY' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'QuadReadAcrossY' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float y = hlsl::QuadReadAcrossY(1.0f); // #QuadReadAcrossY

    // expected-error@#QuadReadAcrossDiagonal {{'QuadReadAcrossDiagonal' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'QuadReadAcrossDiagonal' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float d = hlsl::QuadReadAcrossDiagonal(1.0f); // #QuadReadAcrossDiagonal

    // expected-error@#QuadReadLaneAt {{'QuadReadLaneAt' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'QuadReadLaneAt' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    float a = hlsl::QuadReadLaneAt(1.0f, 0); // #QuadReadLaneAt

    // expected-error@#QuadAll {{'QuadAll' is only available on Shader Model 6.0 or newer}}
    // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* {{'QuadAll' has been marked as being introduced in Shader Model 6.0 here, but the deployment target is Shader Model 5.0}}
    bool qall = hlsl::QuadAll(true); // #QuadAll
}
