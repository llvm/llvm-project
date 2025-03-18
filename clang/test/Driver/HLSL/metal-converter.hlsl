// RUN: %clang_dxc -T cs_6_0 %s -metal -Fo %t.mtl -### 2>&1 | FileCheck %s
// RUN: %clang_dxc -T cs_6_0 %s -metal -Vd -Fo %t.mtl -### 2>&1 | FileCheck %s
// CHECK: "{{.*}}metal-shaderconverter{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.mtl"

// RUN: %clang_dxc -T cs_6_0 %s -metal -### 2>&1 | FileCheck --check-prefix=NO_MTL %s
// NO_MTL-NOT: metal-shaderconverter

// RUN: echo "dxv" > %T/dxv && chmod 754 %T/dxv
// RUN: %clang_dxc -T cs_6_0 %s --dxv-path=%T -metal -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=DXV %s
// DXV: "{{.*}}dxv{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.dxo"
// DXV: "{{.*}}metal-shaderconverter{{(.exe)?}}" "{{.*}}.dxo" "-o" "{{.*}}.mtl"

RWBuffer<float4> In : register(u0, space0);
RWBuffer<float4> Out : register(u1, space4);

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
  Out[GI] = In[GI] * In[GI];
}
