// RUN: %clang_cc1 -finclude-default-header -fhlsl-strict-availability \
// RUN:   -triple dxil-pc-shadermodel6.5-library -verify %s

export void testUnavailable(unsigned Index) {
  
  // expected-error@+2 {{'ResourceDescriptorHeap' is only available on Shader Model 6.6 or newer}}
  // expected-note@hlsl/hlsl_resources.h:* {{'ResourceDescriptorHeap' has been marked as being introduced in Shader Model 6.6 here, but the deployment target is Shader Model 6.5}}
  RWBuffer<int> Buffer = ResourceDescriptorHeap[Index];
  
  // expected-error@+2 {{'SamplerDescriptorHeap' is only available on Shader Model 6.6 or newer}}
  // expected-note@hlsl/hlsl_resources.h:* {{'SamplerDescriptorHeap' has been marked as being introduced in Shader Model 6.6 here, but the deployment target is Shader Model 6.5}}
  SamplerState Sampler = SamplerDescriptorHeap[Index];
}
