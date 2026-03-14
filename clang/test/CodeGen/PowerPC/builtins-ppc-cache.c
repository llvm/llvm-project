// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -o - %s | FileCheck %s

int A;
int B[5];
float C;
float D[5];
double E;
double F[5];

void func(int a, int b[], float c, float d[], double e, double f[]) {
  __builtin_dcbf (&a);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&A);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&b[2]);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&B[2]);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&c);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&C);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&d[2]);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&D[2]);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&e);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&E);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&f[0]);
  // CHECK: @llvm.ppc.dcbf(ptr

  __builtin_dcbf (&F[0]);
  // CHECK: @llvm.ppc.dcbf(ptr
}
