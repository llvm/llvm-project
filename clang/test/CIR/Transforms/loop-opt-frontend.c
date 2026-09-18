// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir --cir-loop-opt=test-annotate=true -o - \
// RUN:   | FileCheck %s --implicit-check-not=cir.loopopt.test

#define N 1024
#define K 64

double A[N][N], B[N][N], C[N][N];
long AL[N][N];

// CHECK-LABEL: cir.func{{.*}} @tri_upper_bound
// CHECK: {cir.loopopt.test = {kind = "inner_affine_upper", offset = 0 : i32, scale = 1 : i32}}
void tri_upper_bound(void) {
  for (int i = 1; i < N; i++)
    for (int j = 0; j < i; j++)
      B[j][i] = A[j][i] * 3.0;
}

// CHECK-LABEL: cir.func{{.*}} @tri_fill
// CHECK: {cir.loopopt.test = {kind = "inner_affine_upper", offset = 0 : i32, scale = 1 : i32}}
void tri_fill(void) {
  for (int i = 1; i < N; i++)
    for (int j = 0; j < i; j++)
      B[j][i] = 0.0;
}

// CHECK-LABEL: cir.func{{.*}} @tri_ldlt_update
// CHECK: {cir.loopopt.test = {kind = "inner_affine_upper", offset = 0 : i32, scale = 1 : i32}}
void tri_ldlt_update(void) {
  for (int i = 1; i < N; i++)
    for (int j = 0; j < i; j++)
      B[j][i] = B[j][i] - A[j][i] * C[j][i];
}

// CHECK-LABEL: cir.func{{.*}} @tri_addk_bound
// CHECK: {cir.loopopt.test = {kind = "inner_affine_upper", offset = 64 : i32, scale = 1 : i32}}
void tri_addk_bound(void) {
  for (int i = 0; i < N - K; i++)
    for (int j = 0; j < i + K; j++)
      B[j][i] = A[j][i] * 3.0;
}

// CHECK-LABEL: cir.func{{.*}} @tri_variant_2i
// CHECK: {cir.loopopt.test = {kind = "inner_affine_upper", offset = 0 : i32, scale = 2 : i32}}
void tri_variant_2i(void) {
  for (int i = 1; i < N / 2; i++)
    for (int j = 0; j < 2 * i; j++)
      B[j][i] = A[j][i] * 3.0;
}

// CHECK-LABEL: cir.func{{.*}} @tri_lower_start
// CHECK: {cir.loopopt.test = {kind = "outer_iv_inner_start"}}
void tri_lower_start(void) {
  for (int i = 0; i < N; i++)
    for (int j = i; j < N; j++)
      B[i][j] = A[j][i] + C[j][i];
}

// CHECK-LABEL: cir.func{{.*}} @tri_mul_bound
// CHECK: {cir.loopopt.test = {kind = "inner_product_upper"}}
void tri_mul_bound(void) {
  for (int i = 1; i < N; i++)
    for (int j = 0; j * i < N; j++)
      B[j][i] = A[j][i] * 3.0;
}

// CHECK-LABEL: cir.func{{.*}} @tri_arg_start
// CHECK: {cir.loopopt.test = {kind = "invariant_inner_start"}}
long tri_arg_start(long lo) {
  long s = 0;
  for (long i = 0; i < N; i++)
    for (long j = lo; j < N; j++)
      s += AL[j][i];
  return s;
}

// CHECK-LABEL: cir.func{{.*}} @rectangular
// CHECK-NOT: cir.loopopt.test
void rectangular(long lo) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      AL[i][j] = lo;
}
