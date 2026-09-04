// RUN: %clang_cc1 -fclangir -emit-cir %s -o - | \
// RUN:   cir-opt -cir-loop-distribution -o /dev/null 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=remark:

double A[8][4], B[4][8], C[4][4], D[4][4];

void candidate(void) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      C[i][j] = 0;
      for (int k = 0; k < 8; ++k)
        C[i][j] += A[k][i] * A[k][j];
    }
}

void perfect_strided_nest(void) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 8; ++k)
        C[i][j] += A[k][i] * A[k][j];
}

void already_unit_stride(void) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      D[i][j] = 0;
      for (int k = 0; k < 8; ++k)
        D[i][j] += B[i][k] * B[j][k];
    }
}

// CHECK: remark: loop distribution candidate in function 'candidate'
