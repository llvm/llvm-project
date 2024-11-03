// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -x c -triple x86_64-apple-darwin10 %s

extern int omp_get_initial_device();
extern void *omp_get_mapped_ptr(void *, int);

void t() {
  omp_get_mapped_ptr(&x, omp_get_initial_device()); //expected-error {{use of undeclared identifier 'x'}}
}

