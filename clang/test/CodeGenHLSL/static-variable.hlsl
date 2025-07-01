// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan1.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=DXIL

// DXIL:  @_ZL1g = internal global float 0.000000e+00, align 4
// SPIRV: @_ZL1g = internal addrspace(10) global float 0.000000e+00, align 4

static float g = 0;

[numthreads(8,8,1)]
void main() {
// DXIL:  {{.*}} = load float, ptr @_ZL1g, align 4
// SPIRV: {{.*}} = load float, ptr addrspace(10) @_ZL1g, align 4
  float l = g;
}
