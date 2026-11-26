// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -finclude-default-header -triple spirv1.6-unknown-vulkan1.3-compute -O3 -emit-obj -o %t.spv %s
// RUN: %if spirv-val %{ spirv-val --target-env vulkan1.3 %t.spv %}

// InstCombine turns the two reads of In into one load through a phi of
// resource pointers, which the SPIR-V backend should compile to a valid module.

RWBuffer<int> In : register(u0);
RWBuffer<int> Out : register(u1);

[numthreads(4, 1, 1)]
void main(uint GI : SV_GroupIndex) {
  int A;
  if (GI == 0)
    A = In[1];
  else
    A = In[2];
  Out[GI] = A;
}
