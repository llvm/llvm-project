// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -fenable-matrix -emit-llvm -O0 -o - %s | FileCheck %s

// A matrix is lowered to a flat vector of the element type, so a matrix of
// `long double` is a vector of real `x86_fp80` values; the backend splits the
// vector and softens every lane.
typedef long double m2x2 __attribute__((matrix_type(2, 2)));
typedef long double m2x3 __attribute__((matrix_type(2, 3)));
typedef long double m3x2 __attribute__((matrix_type(3, 2)));
typedef float f2x3 __attribute__((matrix_type(2, 3)));

// Whole vectors move around as they are.
m2x3 copy(m2x3 a) { return a; }
// CHECK-LABEL: define {{.*}} @copy(
// CHECK-NOT: call
// CHECK: ret <6 x x86_fp80>

m2x3 add(m2x3 a, m2x3 b) { return a + b; }
// CHECK-LABEL: define {{.*}} @add(
// CHECK: fadd <6 x x86_fp80>
// CHECK-NOT: __arm64ec_

m2x3 sub(m2x3 a, m2x3 b) { return a - b; }
// CHECK-LABEL: define {{.*}} @sub(
// CHECK: fsub <6 x x86_fp80>
// CHECK-NOT: __arm64ec_

// A scalar operand of a matrix is splatted into every lane.
m2x3 mul_scalar(m2x3 a, long double s) { return a * s; }
// CHECK-LABEL: define {{.*}} @mul_scalar(
// CHECK: fmul <6 x x86_fp80> {{.*}}, %scalar.splat.splat
// CHECK-NOT: __arm64ec_

m2x3 scalar_mul(long double s, m2x3 a) { return s * a; }
// CHECK-LABEL: define {{.*}} @scalar_mul(
// CHECK: fmul <6 x x86_fp80> %scalar.splat.splat, {{.*}}
// CHECK-NOT: __arm64ec_

m2x3 div_scalar(m2x3 a, long double s) { return a / s; }
// CHECK-LABEL: define {{.*}} @div_scalar(
// CHECK: fdiv <6 x x86_fp80> {{.*}}, %scalar.splat.splat
// CHECK-NOT: __arm64ec_

// Matrix multiplication stays a vector-of-f80 operation.
m2x2 mul_matrix(m2x3 a, m3x2 b) { return a * b; }
// CHECK-LABEL: define {{.*}} @mul_matrix(
// CHECK: %[[P:.*]] = call <4 x x86_fp80> @llvm.matrix.multiply.v4f80.v6f80.v6f80(<6 x x86_fp80> {{.*}}, <6 x x86_fp80> {{.*}}, i32 2, i32 3, i32 2)
// CHECK: ret <4 x x86_fp80> %[[P]]

void mul_assign(m2x2 *a, m2x2 b) { *a *= b; }
// CHECK-LABEL: define {{.*}} @mul_assign(
// CHECK: call <4 x x86_fp80> @llvm.matrix.multiply.v4f80.v4f80.v4f80(<4 x x86_fp80> {{.*}}, <4 x x86_fp80> {{.*}}, i32 2, i32 2, i32 2)
// CHECK: store <4 x x86_fp80>

// Conversions stay vector conversions.
f2x3 truncate(m2x3 a) { return (f2x3)a; }
// CHECK-LABEL: define {{.*}} @truncate(
// CHECK: fptrunc <6 x x86_fp80> {{.*}} to <6 x float>

m2x3 extend(f2x3 a) { return (m2x3)a; }
// CHECK-LABEL: define {{.*}} @extend(
// CHECK: fpext <6 x float> {{.*}} to <6 x x86_fp80>

// Transposing only moves whole values around.
m3x2 transpose(m2x3 a) { return __builtin_matrix_transpose(a); }
// CHECK-LABEL: define {{.*}} @transpose(
// CHECK-NOT: call double
// CHECK: %[[T:.*]] = call <6 x x86_fp80> @llvm.matrix.transpose.v6f80(<6 x x86_fp80>
// CHECK: ret <6 x x86_fp80> %[[T]]
