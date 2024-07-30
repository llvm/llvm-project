// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.5-library -x hlsl %s -verify

[shader("compute")]
[numthreads(1,1,1)]
#if __SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR == 6
// expected-error@+4 {{wavesize only takes multiple arguments in shader model 6.8 or higher}}
#elif __SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR == 5
// expected-error@+2 {{attribute wavesize requires shader model 6.6 or greater}}
#endif
[WaveSize(4, 16)]
void e0() {
}

[shader("compute")]
[numthreads(1,1,1)]
#if __SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR == 6
// expected-error@+4 {{wavesize only takes multiple arguments in shader model 6.8 or higher}}
#elif __SHADER_TARGET_MAJOR == 6 && __SHADER_TARGET_MINOR == 5
// expected-error@+2 {{attribute wavesize requires shader model 6.6 or greater}}
#endif
[WaveSize(4, 16)]
void e1() {
}
