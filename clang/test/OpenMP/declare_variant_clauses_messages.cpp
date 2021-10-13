// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -DOMP51 -std=c++11 -o - %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -DOMP50 -std=c++11 -o - %s

int Other;

void foo_v1(float *AAA, float *BBB, int *I) { return; }
void foo_v2(float *AAA, float *BBB, int *I) { return; }
void foo_v3(float *AAA, float *BBB, int *I) { return; }

#ifdef OMP51
// expected-error@+3 {{'adjust_arg' argument 'AAA' used in multiple clauses}}
#pragma omp declare variant(foo_v1)                          \
   match(construct={dispatch}, device={arch(arm)})           \
   adjust_args(need_device_ptr:AAA,BBB) adjust_args(need_device_ptr:AAA)

// expected-error@+3 {{'adjust_arg' argument 'AAA' used in multiple clauses}}
#pragma omp declare variant(foo_v1)                          \
   match(construct={dispatch}, device={arch(ppc)}),          \
   adjust_args(need_device_ptr:AAA) adjust_args(nothing:AAA)

// expected-error@+2 {{use of undeclared identifier 'J'}}
#pragma omp declare variant(foo_v1)                          \
   adjust_args(nothing:J)                                    \
   match(construct={dispatch}, device={arch(x86,x86_64)})

// expected-error@+2 {{expected reference to one of the parameters of function 'foo'}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:Other)                                \
   match(construct={dispatch}, device={arch(x86,x86_64)})

// expected-error@+2 {{'adjust_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:BBB) match(construct={target}, device={arch(arm)})

// expected-error@+2 {{'adjust_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:BBB) match(device={arch(ppc)})
#endif // OMP51
#ifdef OMP50
// expected-error@+2 {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo_v1)                            \
   adjust_args(need_device_ptr:AAA) match(device={arch(arm)})
#endif // OMP50

void foo(float *AAA, float *BBB, int *I) { return; }

