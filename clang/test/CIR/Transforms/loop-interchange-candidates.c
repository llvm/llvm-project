// RUN: %clang_cc1 -fclangir -emit-cir %s -o - | cir-opt -cir-loop-interchange -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=remark:

double A[256][256], x[256], y[256];

void column_strided(void) {
  for (int i = 0; i < 256; i++)
    for (int j = 0; j < 256; j++)
      x[i] += A[j][i] * y[j];
}
// CHECK: remark: loop interchange candidate in function 'column_strided'

void row_contiguous(void) {
  for (int i = 0; i < 256; i++)
    for (int j = 0; j < 256; j++)
      x[i] += A[i][j] * y[j];
}

void not_perfect(void) {
  for (int i = 0; i < 256; i++) {
    x[i] = 0;
    for (int j = 0; j < 256; j++)
      x[i] += A[j][i] * y[j];
  }
}

void triangular(void) {
  for (int i = 0; i < 256; i++)
    for (int j = 0; j < i; j++)
      x[i] += A[j][i] * y[j];
}

void not_less_than(void) {
  for (int i = 0; i < 256; i++)
    for (int j = 0; j <= 255; j++)
      x[i] += A[j][i] * y[j];
}
