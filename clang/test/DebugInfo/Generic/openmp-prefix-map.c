// RUN: %clang_cc1 -triple i386-unknown-unknown -debug-info-kind=standalone -fopenmp %s -emit-llvm -o - -disable-llvm-optzns -fdebug-prefix-map=%S=.| FileCheck -DPREFIX=%S %s

// CHECK-NOT: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";[[PREFIX]]{{.*}}.c;foo;{{[0-9]+}};{{[0-9]+}};;\00"

void work1(int, int);
void work2(int, int);
void work12(int, int);

void foo(int q) {
  int p = 2;

  #pragma omp parallel firstprivate(q, p)
  work1(p, q);

  #pragma omp parallel for firstprivate(p, q)
  for (int i = 0; i < q; i++)
    work2(i, p);

  #pragma omp target teams firstprivate(p)
  work12(p, p);
}
