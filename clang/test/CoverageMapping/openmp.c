// RUN: %clang_cc1 -fopenmp -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name openmp.c %s | FileCheck %s

// CHECK: openmp.c:{{.+}}omp_outlined{{.+}}:
// CHECK: File 0, 10:3 -> 10:31
// CHECK: File 0, 10:19 -> 10:24
// CHECK: File 0, 10:26 -> 10:29
// CHECK: File 0, 10:30 -> 10:31
int foo(int time, int n) {
#pragma omp parallel for default(shared) schedule(dynamic, 1) reduction(+ : time)
  for (int i = 1; i < n; ++i);
  return 0;
}
