// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library -verify %s

void useBuffer(RWBuffer<int> Buffer) {}
void useSampler(SamplerState Sampler) {}

export void test(unsigned Index) {
  
  // valid
  RWBuffer<int> Buf1 = ResourceDescriptorHeap[Index];
  Buf1 = ResourceDescriptorHeap[Index + 1];
  useBuffer(ResourceDescriptorHeap[Index + 2]);

  // valid
  SamplerState Sampler = SamplerDescriptorHeap[Index];
  Sampler = SamplerDescriptorHeap[Index + 1];
  useSampler(SamplerDescriptorHeap[Index + 2]);

  // expected-error@+3 {{no viable conversion from '__hlsl_heap_sampler_info' to 'RWBuffer<int>'}}
  // expected-note@*:* {{candidate constructor not viable: no known conversion from '__hlsl_heap_sampler_info' to 'const hlsl::RWBuffer<int> &' for 1st argument}}
  // expected-note@*:* {{candidate constructor not viable: no known conversion from '__hlsl_heap_sampler_info' to 'hlsl::__hlsl_heap_resource_info' for 1st argument}}
  RWBuffer<int> Buf2 = SamplerDescriptorHeap[Index];

  // expected-error@+3 {{no viable conversion from '__hlsl_heap_resource_info' to 'SamplerState'}}
  // expected-note@*:* {{candidate constructor not viable: no known conversion from '__hlsl_heap_resource_info' to 'const hlsl::SamplerState &' for 1st argument}}
  // expected-note@*:* {{candidate constructor not viable: no known conversion from '__hlsl_heap_resource_info' to 'hlsl::__hlsl_heap_sampler_info' for 1st argument}}
  SamplerState Sampler2 = ResourceDescriptorHeap[Index];
  
  // expected-error@+2 {{no viable overloaded '='}}
  // expected-note@*:* {{candidate function not viable: no known conversion from '__hlsl_heap_sampler_info' to 'RWBuffer<int>' for 1st argument}}
  Buf2 = SamplerDescriptorHeap[Index];
  
  // expected-error@+2 {{no viable overloaded '='}}
  // expected-note@*:* {{candidate function not viable: no known conversion from '__hlsl_heap_resource_info' to 'SamplerState' for 1st argument}}
  Sampler2 = ResourceDescriptorHeap[Index];
  
  // expected-error@+2 {{no matching function for call to 'useBuffer'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from '__hlsl_heap_sampler_info' to 'const hlsl::RWBuffer<int>' for 1st argument}}
  useBuffer(SamplerDescriptorHeap[Index]);
  
  // expected-error@+2 {{no matching function for call to 'useSampler'}}
  // expected-note@*:* {{candidate function not viable: no known conversion from '__hlsl_heap_resource_info' to 'const hlsl::SamplerState' for 1st argument}}
  useSampler(ResourceDescriptorHeap[Index]);
  
  // expected-error@+1 {{no member named 'Load'}}
  ResourceDescriptorHeap[Index].Load(0);
  
  // expected-error@+1 {{no member named 'Sample'}}
  SamplerDescriptorHeap[Index].Sample(0);

}
