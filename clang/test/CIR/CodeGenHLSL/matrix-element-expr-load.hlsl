// RUN: not %clang_cc1 -x hlsl -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:   -fclangir -emit-cir -disable-llvm-passes 2>&1 | FileCheck %s

// CHECK: ClangIR code gen Not Yet Implemented: processing of type: ConstantMatrix
float1 test_zero_indexed(float2x2 M) { 
  // CHECK: ClangIR code gen Not Yet Implemented: ScalarExprEmitter: matrix element
  return M._m00; 
}
