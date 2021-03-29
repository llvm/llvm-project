// REQUIRES: x86-registered-target
// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -Rpass-analysis=loop-vectorize -emit-llvm -S %s -o - 2>&1 | FileCheck %s

// CHECK: {{.*}}:9:11: remark: loop not vectorized: cannot prove it is safe to reorder floating-point operations; allow reordering by specifying '#pragma clang loop vectorize(enable)' before the loop or by providing the compiler option '-ffast-math'.

double foo(int N) {
  double v = 0.0;

  for (int i = 0; i < N; i++)
    v = v + 1.0;

  return v;
}

// CHECK: {{.*}}:17:3: remark: loop not vectorized: cannot prove it is safe to reorder memory operations; allow reordering by specifying '#pragma clang loop vectorize(enable)' before the loop. If the arrays will always be independent specify '#pragma clang loop vectorize(assume_safety)' before the loop or provide the '__restrict__' qualifier with the independent array arguments. Erroneous results will occur if these options are incorrectly applied!

void foo2(int *dw, int *uw, int *A, int *B, int *C, int *D, int N) {
  for (long i = 0; i < N; i++) {
    dw[i] = A[i] + B[i - 1] + C[i - 2] + D[i - 3];
    uw[i] = A[i] + B[i + 1] + C[i + 2] + D[i + 3];
  }
}
