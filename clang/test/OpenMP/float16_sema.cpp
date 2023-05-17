// RUN: %clang_cc1 -fsyntax-only -x c++ -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=nvptx64 -verify %s
// expected-no-diagnostics

int foo() {
#pragma omp target
  {
    __fp16 a = -1.0f16;
  }
  return 0;
}
