// RUN: mkdir -p %t.dir
// RUN: echo "dxv" > %t.dir/dxv && chmod 754 %t.dir/dxv

// RUN: env PATH="" %clang_dxc -T cs_6_0 %s -metal -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=NO_DXV %s
// RUN: env PATH="" %clang_dxc -T cs_6_0 %s -metal -Vd -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=NO_DXV %s
// RUN: env PATH="" %clang_dxc -T cs_6_0 %s --dxv-path=%t.dir -metal -Vd -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=NO_DXV %s
// NO_DXV: "{{.*}}metal-shaderconverter{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.mtl"

// RUN: env PATH="" %clang_dxc -T cs_6_0 %s -metal -Fre blah.json -Vd -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=FRE %s
// FRE: "{{.*}}metal-shaderconverter{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.mtl" "--output-reflection-file=blah.json"

// RUN: env PATH="" not %clang_dxc -T cs_6_0 %s -metal -Fre blah.json -Vd -### 2>&1 | FileCheck --check-prefix=FRE_ERR %s
// FRE_ERR: error: -Fre option requires -Fo option when targeting Metal

// Does not generate the metal IR when the output file is not specified since we
// cannot disassemble the metal IR reliably.
// RUN: %clang_dxc -T cs_6_0 %s -metal -### 2>&1 | FileCheck --check-prefix=NO_MTL %s
// NO_MTL-NOT: metal-shaderconverter

// RUN: %clang_dxc -T cs_6_0 %s --dxv-path=%t.dir -metal -Fo %t.mtl -### 2>&1 | FileCheck --check-prefix=DXV %s
// DXV: "{{.*}}dxv{{(.exe)?}}" "{{.*}}.obj" "-o" "{{.*}}.dxo"
// DXV: "{{.*}}metal-shaderconverter{{(.exe)?}}" "{{.*}}.dxo" "-o" "{{.*}}.mtl"

RWBuffer<float4> In : register(u0, space0);
RWBuffer<float4> Out : register(u1, space4);

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
  Out[GI] = In[GI] * In[GI];
}
