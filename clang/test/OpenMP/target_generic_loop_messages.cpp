// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -fopenmp \
// RUN:  -fopenmp-version=60 -Wuninitialized %s

void foo()
{
  int i,j,k;
  int z;

  // expected-error@+2 {{statement after '#pragma omp target loop' must be a for loop}}
  #pragma omp target loop bind(thread)
  i = 0;

  // OpenMP 5.1 [2.22 Nesting of regions]
  //
  // A barrier region may not be closely nested inside a worksharing, loop,
  // task, taskloop, critical, ordered, atomic, or masked region.

  // expected-error@+3 {{region cannot be closely nested inside 'target loop' region}}
  #pragma omp target loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp barrier
  }

  // A masked region may not be closely nested inside a worksharing, loop,
  // atomic, task, or taskloop region.

  // expected-error@+3 {{region cannot be closely nested inside 'target loop' region}}
  #pragma omp target loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp masked filter(2)
    { }
  }

  // An ordered region that corresponds to an ordered construct without any
  // clause or with the threads or depend clause may not be closely nested
  // inside a critical, ordered, loop, atomic, task, or taskloop region.

  // expected-error@+3 {{region cannot be closely nested inside 'target loop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
  #pragma omp target loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp ordered
    { }
  }

  // expected-error@+3 {{region cannot be closely nested inside 'target loop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
  #pragma omp target loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp ordered threads
    { }
  }

  // bind clause

  // expected-error@+1 {{directive '#pragma omp target loop' cannot contain more than one 'bind' clause}}
  #pragma omp target loop bind(thread) bind(thread)
  for (i=0; i<1000; ++i) {
  }

  // expected-error@+1 {{expected 'teams', 'parallel' or 'thread' in OpenMP clause 'bind'}}
  #pragma omp target loop bind(other)
  for (i=0; i<1000; ++i) {
  }

  // expected-error@+1 {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
  #pragma omp target loop bind(teams)
  for (i=0; i<1000; ++i) {
  }

  // collapse clause

  // expected-error@+4 {{expected 2 for loops after '#pragma omp target loop', but found only 1}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp target loop collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{directive '#pragma omp target loop' cannot contain more than one 'collapse' clause}}
  #pragma omp target loop collapse(2) collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    for (j=0; j<1000; ++j)
      z = i+j+11;

  // order clause

  // expected-error@+1 {{expected 'concurrent' in OpenMP clause 'order'}}
  #pragma omp target loop order(foo) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // private clause

  // expected-error@+1 {{use of undeclared identifier 'undef_var'}}
  #pragma omp target loop private(undef_var) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // lastprivate

  // A list item may not appear in a lastprivate clause unless it is the loop
  // iteration variable of a loop that is associated with the construct.

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp target loop' directives}}
  #pragma omp target loop lastprivate(z) bind(thread)
  for (i=0; i<1000; ++i) {
    z = i+11;
  }

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp target loop' directives}}
  #pragma omp target loop lastprivate(k) collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    for (j=0; j<1000; ++j)
      for (k=0; k<1000; ++k)
        z = i+j+k+11;

  // reduction

  // expected-error@+1 {{use of undeclared identifier 'undef_var'}}
  #pragma omp target loop reduction(+:undef_var) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // nowait

  // expected-error@+1 {{directive '#pragma omp target loop' cannot contain more than one 'nowait' clause}}
  #pragma omp target loop nowait nowait
  for (i=0; i<1000; ++i)
    z = i+11;

  // thread_limit

  // expected-error@+1 {{directive '#pragma omp target loop' cannot contain more than one 'thread_limit' clause}}
  #pragma omp target loop thread_limit(4) thread_limit(4)
  for (i=0; i<1000; ++i)
    z = i+11;

  // Clauses of 'target parallel loop' that 'target loop' does not accept.

  // expected-error@+1 {{unexpected OpenMP clause 'num_threads' in directive '#pragma omp target loop'}}
  #pragma omp target loop num_threads(4)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{unexpected OpenMP clause 'proc_bind' in directive '#pragma omp target loop'}}
  #pragma omp target loop proc_bind(close)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{unexpected OpenMP clause 'shared' in directive '#pragma omp target loop'}}
  #pragma omp target loop shared(z)
  for (i=0; i<1000; ++i)
    z = i+11;
}

template <typename T, int C>
void templ_test(T t) {
  T i,z;

  // expected-error@+4 {{expected 2 for loops after '#pragma omp target loop', but found only 1}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp target loop collapse(C) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp target loop' directives}}
  #pragma omp target loop lastprivate(z) bind(thread)
  for (i=0; i<1000; ++i) {
    z = i+11;
  }
}

void bar()
{
  templ_test<int, 2>(16); // expected-note {{in instantiation of function template specialization 'templ_test<int, 2>' requested here}}
}
