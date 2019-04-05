// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: #pragma omp declare reduction (+ : int : omp_out *= omp_in){{$}}
// CHECK-NEXT: #pragma omp declare reduction (+ : char : omp_out *= omp_in)

#pragma omp declare reduction(fun : float : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)
// CHECK: #pragma omp declare reduction (fun : float : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)

// CHECK: struct SSS {
struct SSS {
  int field;
#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
  // CHECK: #pragma omp declare reduction (+ : int : omp_out *= omp_in)
  // CHECK-NEXT: #pragma omp declare reduction (+ : char : omp_out *= omp_in)
};
// CHECK: };

void init(struct SSS *priv, struct SSS orig);

#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
// CHECK: #pragma omp declare reduction (fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))

// CHECK: int main() {
int main() {
#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
  // CHECK: #pragma omp declare reduction (fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
  {
#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
  // CHECK: #pragma omp declare reduction (fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
  }
  return 0;
}
// CHECK: }

#pragma omp declare reduction(mymin:int                                        \
                              : omp_out = omp_out > omp_in ? omp_in : omp_out) \
    initializer(omp_priv = 2147483647)

int foo(int argc, char **argv) {
  int x;
#pragma omp parallel for reduction(mymin : x)
  for (int i = 0; i < 1000; i++)
    ;
  return 0;
}

// CHECK: #pragma omp parallel for reduction(mymin: x)
#endif
