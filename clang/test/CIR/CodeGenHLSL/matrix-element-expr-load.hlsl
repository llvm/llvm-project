// RUN: %clang_cc1 -x hlsl -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:   -fclangir -emit-cir -disable-llvm-passes -verify

// expected-error@*:* {{ClangIR code gen Not Yet Implemented: processing of type: ConstantMatrix}}
// expected-error@*:* {{ClangIR code gen Not Yet Implemented: Matrix type conversion}}
float test_zero_indexed(float2x2 M) {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: ScalarExprEmitter: matrix element}}
  return M._m00;
}
