// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - \
// RUN:   | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s --check-prefix=DXIL

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
