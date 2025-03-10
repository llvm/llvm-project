// RUN: %clang_dxc -T cs_6_0 %s -metal -Fo tmp.mtl -### 2>&1 | FileCheck %s
// RUN: %clang_dxc -T cs_6_0 %s -metal -Vd -Fo tmp.mtl -### 2>&1 | FileCheck %s
// CHECK: "{{.*}}metal-shaderconverter{{(.exe)?}}" "tmp.mtl" "-o" "tmp.mtl"

// RUN: %clang_dxc -T cs_6_0 %s -metal -### 2>&1 | FileCheck --check-prefix=NO_MTL %s
// NO_MTL-NOT: metal-shaderconverter

RWBuffer<float4> In : register(u0, space0);
RWBuffer<float4> Out : register(u1, space4);

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
  Out[GI] = In[GI] * In[GI];
}
