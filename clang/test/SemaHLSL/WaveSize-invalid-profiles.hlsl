// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-pixel -x hlsl %s  -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -x hlsl %s  -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-geometry -x hlsl %s  -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-hull -x hlsl %s  -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-domain -x hlsl %s  -verify

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
// expected-error@#WaveSize {{attribute 'WaveSize' is unsupported in 'pixel' shaders, requires one of the following: compute, amplification, mesh}}
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
// expected-error@#WaveSize {{attribute 'WaveSize' is unsupported in 'vertex' shaders, requires one of the following: compute, amplification, mesh}}
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_GEOMETRY
// expected-error@#WaveSize {{attribute 'WaveSize' is unsupported in 'geometry' shaders, requires one of the following: compute, amplification, mesh}}
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_HULL
// expected-error@#WaveSize {{attribute 'WaveSize' is unsupported in 'hull' shaders, requires one of the following: compute, amplification, mesh}}
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_DOMAIN
// expected-error@#WaveSize {{attribute 'WaveSize' is unsupported in 'domain' shaders, requires one of the following: compute, amplification, mesh}}
#endif
[WaveSize(16)] // #WaveSize
void main() {
}
