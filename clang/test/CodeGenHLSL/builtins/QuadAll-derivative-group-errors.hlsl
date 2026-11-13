// RUN: not %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o %t 2>&1 | FileCheck %s \
// RUN:   --implicit-check-not="entry point 'valid" \
// RUN:   --implicit-check-not="entry point 'no_vote'"
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o %t

bool helper(bool cond) {
  return QuadAll(cond);
}

[shader("compute")]
[numthreads(4, 1, 1)]
void valid_linear() {
  helper(true);
}

[shader("compute")]
[numthreads(2, 2, 3)]
void valid_quads() {
  helper(true);
}

[shader("compute")]
[numthreads(6, 1, 1)]
void invalid_linear() {
  (void)__builtin_hlsl_quad_all(true);
}

[shader("compute")]
[numthreads(3, 4, 1)]
void invalid_quad_x() {
  helper(true);
}

[shader("compute")]
[numthreads(4, 3, 1)]
void invalid_quad_y() {
  helper(true);
}

[shader("compute")]
[numthreads(2, 1, 2)]
void invalid_quad_3d() {
  helper(true);
}

[shader("compute")]
[numthreads(3, 1, 1)]
void no_vote() {}

// CHECK-DAG: error: entry point 'invalid_linear' uses QuadAll with invalid numthreads: DerivativeGroupLinearKHR requires X * Y * Z to be a multiple of 4
// CHECK-DAG: error: entry point 'invalid_quad_x' uses QuadAll with invalid numthreads: DerivativeGroupQuadsKHR requires X and Y to be multiples of 2
// CHECK-DAG: error: entry point 'invalid_quad_y' uses QuadAll with invalid numthreads: DerivativeGroupQuadsKHR requires X and Y to be multiples of 2
// CHECK-DAG: error: entry point 'invalid_quad_3d' uses QuadAll with invalid numthreads: DerivativeGroupQuadsKHR requires X and Y to be multiples of 2
