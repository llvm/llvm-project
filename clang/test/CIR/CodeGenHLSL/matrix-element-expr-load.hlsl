// RUN: not %clang_cc1 -x hlsl -finclude-default-header -triple spirv-unknown-vulkan-library %s \
// RUN:   -fclangir -emit-cir -disable-llvm-passes -verify

float test_zero_indexed(float2x2 M) {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: ScalarExprEmitter: matrix element}}
  return M._m00;
}
