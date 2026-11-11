// DEFINE: %{deriv-ext} = +SPV_KHR_compute_shader_derivatives
// DEFINE: %{quad-exts} = +SPV_KHR_quad_control,+SPV_KHR_maximal_reconvergence
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - \
// RUN:   | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s --check-prefix=DXIL
// RUN: %if spirv-registered-target %{ \
// RUN:   %clang_cc1 -finclude-default-header -triple \
// RUN:     spirv1.6-unknown-vulkan1.3-library -O3 -S %s -o - \
// RUN:     -mllvm -spirv-ext=%{deriv-ext},%{quad-exts} \
// RUN:     | FileCheck %s --check-prefixes=SPIRV,NATIVE %}
// RUN: %if spirv-registered-target %{ \
// RUN:   %clang_cc1 -finclude-default-header -triple \
// RUN:     spirv1.6-unknown-vulkan1.3-library -O3 -S %s -o - \
// RUN:     -mllvm -spirv-ext=%{deriv-ext},-SPV_KHR_quad_control \
// RUN:     | FileCheck %s --check-prefixes=SPIRV,EMULATED \
// RUN:       --implicit-check-not=OpGroupNonUniformQuadAllKHR %}

RWStructuredBuffer<uint> Out;

bool vote(bool cond) {
  return QuadAll(cond);
}

bool helper(bool cond) {
  return vote(cond);
}

export bool exported_vote(bool cond) {
  return helper(cond);
}

[shader("compute")]
[numthreads(4, 2, 1)]
void compute_2d(uint index : SV_GroupIndex) {
  Out[index] = helper(index != 0) && helper(index < 8);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void compute_1d(uint index : SV_GroupIndex) {
  Out[index] = __builtin_hlsl_quad_all(index != 0);
}

[shader("pixel")]
void pixel_main() {
  Out[0] = helper(Out[0] != 0);
}

[shader("compute")]
[numthreads(4, 2, 1)]
void no_vote(uint index : SV_GroupIndex) {
  Out[index] = index;
}

// IR: !spirv.ExecutionMode = !{!{{[0-9]+}}, !{{[0-9]+}}}
// IR-DAG: !{{[0-9]+}} = !{ptr @compute_2d, i32 5289}
// IR-DAG: !{{[0-9]+}} = !{ptr @compute_1d, i32 5290}
// DXIL-NOT: !spirv.ExecutionMode

// SPIRV-DAG: OpCapability ComputeDerivativeGroupQuadsKHR
// SPIRV-DAG: OpCapability ComputeDerivativeGroupLinearKHR
// SPIRV-DAG: OpExtension "SPV_KHR_compute_shader_derivatives"
// SPIRV-DAG: OpEntryPoint GLCompute %[[#QUADS:]] "compute_2d"
// SPIRV-DAG: OpEntryPoint GLCompute %[[#LINEAR:]] "compute_1d"
// SPIRV-DAG: OpExecutionMode %[[#QUADS]] DerivativeGroupQuadsKHR
// SPIRV-DAG: OpExecutionMode %[[#LINEAR]] DerivativeGroupLinearKHR
// NATIVE: OpGroupNonUniformQuadAllKHR
// EMULATED: OpGroupNonUniformQuadSwap
