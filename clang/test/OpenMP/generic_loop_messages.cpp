// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -verify -fopenmp \
// RUN:  -fopenmp-version=51 -Wuninitialized %s

void foo()
{
  int i,j,k;
  int z;

  // expected-error@+2 {{statement after '#pragma omp loop' must be a for loop}}
  #pragma omp loop bind(thread)
  i = 0;

  // OpenMP 5.1 [2.22 Nesting of regions]
  //
  // A barrier region may not be closely nested inside a worksharing, loop,
  // task, taskloop, critical, ordered, atomic, or masked region.

  // expected-error@+3 {{region cannot be closely nested inside 'loop' region}}
  #pragma omp loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp barrier
  }

  // A masked region may not be closely nested inside a worksharing, loop,
  // atomic, task, or taskloop region.

  // expected-error@+3 {{region cannot be closely nested inside 'loop' region}}
  #pragma omp loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp masked filter(2)
    { }
  }

  // An ordered region that corresponds to an ordered construct without any
  // clause or with the threads or depend clause may not be closely nested
  // inside a critical, ordered, loop, atomic, task, or taskloop region.

  // expected-error@+3 {{region cannot be closely nested inside 'loop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
  #pragma omp loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp ordered
    { }
  }

  // expected-error@+3 {{region cannot be closely nested inside 'loop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
  #pragma omp loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp ordered threads
    { }
  }

  // expected-error@+3 {{region cannot be closely nested inside 'loop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
  #pragma omp loop bind(thread)
  for (i=0; i<1000; ++i) {
    #pragma omp ordered depend(source)
  }

  // bind clause

  // expected-error@+1 {{directive '#pragma omp loop' cannot contain more than one 'bind' clause}}
  #pragma omp loop bind(thread) bind(thread)
  for (i=0; i<1000; ++i) {
  }

  // expected-error@+2 {{expected 'teams', 'parallel' or 'thread' in OpenMP clause 'bind'}}
  #pragma omp parallel
  #pragma omp loop bind(other)
  for (i=0; i<1000; ++i) {
  }

  #pragma omp target
  {
    // expected-error@+1 {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    #pragma omp loop bind(teams)
    for (i=0; i<10; ++i) {
    }
  }

  // collapse clause

  // expected-error@+4 {{expected 2 for loops after '#pragma omp loop', but found only 1}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp loop collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{directive '#pragma omp loop' cannot contain more than one 'collapse' clause}}
  #pragma omp loop collapse(2) collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    for (j=0; j<1000; ++j)
      z = i+j+11;

  // order clause

  // expected-error@+1 {{expected 'concurrent' in OpenMP clause 'order'}}
  #pragma omp loop order(foo) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // private clause

  // expected-error@+1 {{use of undeclared identifier 'undef_var'}}
  #pragma omp loop private(undef_var) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // lastprivate

  // A list item may not appear in a lastprivate clause unless it is the loop
  // iteration variable of a loop that is associated with the construct.

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp loop' directives}}
  #pragma omp loop lastprivate(z) bind(thread)
  for (i=0; i<1000; ++i) {
    z = i+11;
  }

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp loop' directives}}
  #pragma omp loop lastprivate(k) collapse(2) bind(thread)
  for (i=0; i<1000; ++i)
    for (j=0; j<1000; ++j)
      for (k=0; k<1000; ++k)
        z = i+j+k+11;

  // reduction

  // expected-error@+1 {{use of undeclared identifier 'undef_var'}}
  #pragma omp loop reduction(+:undef_var) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;
}

template <typename T, int C>
void templ_test(T t) {
  T i,z;

  // expected-error@+4 {{expected 2 for loops after '#pragma omp loop', but found only 1}}
  // expected-note@+1 {{as specified in 'collapse' clause}}
  #pragma omp loop collapse(C) bind(thread)
  for (i=0; i<1000; ++i)
    z = i+11;

  // expected-error@+1 {{only loop iteration variables are allowed in 'lastprivate' clause in 'omp loop' directives}}
  #pragma omp loop lastprivate(z) bind(thread)
  for (i=0; i<1000; ++i) {
    z = i+11;
  }
}

void bar()
{
  templ_test<int, 2>(16); // expected-note {{in instantiation of function template specialization 'templ_test<int, 2>' requested here}}
}
