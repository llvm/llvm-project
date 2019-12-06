// RUN: %clang_cc1 %s -opt-record-file=t1.opt -fopenmp -emit-llvm-bc -o %t.bc
// RUN: %clang_cc1 -x ir %t.bc -opt-record-file %t.opt -fopenmp -emit-obj
// RUN: cat %t.opt |  FileCheck -check-prefix=CHECK  %s

void foo(int *a, int *b, int *c) {
#pragma omp parallel for
 for (int i = 0; i < 100; i++) {
 a[i] = b[i] + c[i];
 }
}

// CHECK: --- !Missed
// CHECK: Pass:            inline
// CHECK: Name:            NoDefinition
// CHECK: Function:        foo
