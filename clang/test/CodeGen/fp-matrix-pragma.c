// RUN: %clang -emit-llvm -S -fenable-matrix -mllvm -disable-llvm-optzns %s -o - | FileCheck %s

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
typedef int ix2x2_t __attribute__((matrix_type(2, 2)));

fx2x2_t fp_matrix_contract(fx2x2_t a, fx2x2_t b, float c, float d) {
// CHECK: call contract <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32
// CHECK: fdiv contract <4 x float>
// CHECK: fmul contract <4 x float>
#pragma clang fp contract(fast)
  return (a * b / c) * d;
}

fx2x2_t fp_matrix_reassoc(fx2x2_t a, fx2x2_t b, fx2x2_t c) {
// CHECK: fadd reassoc <4 x float>
// CHECK: fsub reassoc <4 x float>
#pragma clang fp reassociate(on)
  return a + b - c;
}

fx2x2_t fp_matrix_ops(fx2x2_t a, fx2x2_t b, fx2x2_t c) {
// CHECK: call reassoc contract <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32
// CHECK: fadd reassoc contract <4 x float>
#pragma clang fp contract(fast) reassociate(on)
  return a * b + c;
}

fx2x2_t fp_matrix_compound_ops(fx2x2_t a, fx2x2_t b, fx2x2_t c, fx2x2_t d,
    float e, float f) {
// CHECK: call reassoc contract <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32
// CHECK: fadd reassoc contract <4 x float>
// CHECK: fsub reassoc contract <4 x float>
// CHECK: fmul reassoc contract <4 x float>
// CHECK: fdiv reassoc contract <4 x float>
#pragma clang fp contract(fast) reassociate(on)
  a *= b;
  a += c;
  a -= d;
  a *= e;
  a /= f;

  return a;
}

ix2x2_t int_matrix_ops(ix2x2_t a, ix2x2_t b, ix2x2_t c) {
// CHECK: call <4 x i32> @llvm.matrix.multiply.v4i32.v4i32.v4i32
// CHECK: add <4 x i32>
#pragma clang fp contract(fast) reassociate(on)
  return a * b + c;
}
