// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -o - -DWITHDEF %s
// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -o - -DWITHOUTDEF %s

#ifdef WITHDEF
typedef void *omp_interop_t;

void foo(int *Ap) {
  omp_interop_t InteropVar;
  omp_interop_t Another;

  //expected-error@+1 {{expected interop type: 'target' and/or 'targetsync'}}
  #pragma omp interop init(target,foo:InteropVar) init(target:Another)

  //expected-error@+1 {{use of undeclared identifier 'NoDeclVar'}}
  #pragma omp interop init(target:NoDeclVar) init(target:Another)

  //expected-error@+1 {{use of undeclared identifier 'NoDeclVar'}}
  #pragma omp interop use(NoDeclVar) use(Another)

  //expected-error@+2 {{expected interop type: 'target' and/or 'targetsync'}}
  //expected-error@+1 {{expected expression}}
  #pragma omp interop init(InteropVar) init(target:Another)

  //expected-warning@+1 {{missing ':' after interop types}}
  #pragma omp interop init(target InteropVar)

  //expected-error@+1 {{expected expression}}
  #pragma omp interop init(prefer_type(1,+,3),target:InteropVar) \
                      init(target:Another)

  int IntVar;
  struct S { int I; } SVar;

  //expected-error@+1 {{interop variable must be of type 'omp_interop_t'}}
  #pragma omp interop init(prefer_type(1,"sycl",3),target:IntVar) \
                      init(target:Another)

  //expected-error@+1 {{interop variable must be of type 'omp_interop_t'}}
  #pragma omp interop use(IntVar) use(Another)

  //expected-error@+1 {{interop variable must be of type 'omp_interop_t'}}
  #pragma omp interop init(prefer_type(1,"sycl",3),target:SVar) \
                      init(target:Another)

  //expected-error@+1 {{interop variable must be of type 'omp_interop_t'}}
  #pragma omp interop use(SVar) use(Another)

  int a, b;
  //expected-error@+1 {{expected variable of type 'omp_interop_t'}}
  #pragma omp interop init(target:a+b) init(target:Another)

  //expected-error@+1 {{expected variable of type 'omp_interop_t'}}
  #pragma omp interop use(a+b) use(Another)

  const omp_interop_t C = (omp_interop_t)5;
  //expected-error@+1 {{expected non-const variable of type 'omp_interop_t'}}
  #pragma omp interop init(target:C) init(target:Another)

  //expected-error@+1 {{prefer_list item must be a string literal or constant integral expression}}
  #pragma omp interop init(prefer_type(1.0),target:InteropVar) \
                      init(target:Another)

  //expected-error@+1 {{prefer_list item must be a string literal or constant integral expression}}
  #pragma omp interop init(prefer_type(a),target:InteropVar) \
                      init(target:Another)

  //expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop device(0)

  //expected-warning@+1 {{interop type 'target' cannot be specified more than once}}
  #pragma omp interop init(target,targetsync,target:InteropVar)

  //expected-error@+1 {{'depend' clause requires the 'targetsync' interop type}}
  #pragma omp interop init(target:InteropVar) depend(inout:Ap)

  //expected-error@+1 {{interop variable 'InteropVar' used in multiple action clauses}}
  #pragma omp interop init(target:InteropVar) init(target:InteropVar)

  //expected-error@+1 {{interop variable 'InteropVar' used in multiple action clauses}}
  #pragma omp interop use(InteropVar) use(InteropVar)

  //expected-error@+1 {{interop variable 'InteropVar' used in multiple action clauses}}
  #pragma omp interop init(target:InteropVar) use(InteropVar)

  //expected-error@+1 {{directive '#pragma omp interop' cannot contain more than one 'device' clause}}
  #pragma omp interop init(target:InteropVar) device(0) device(1)

  //expected-error@+1 {{argument to 'device' clause must be a non-negative integer value}}
  #pragma omp interop init(target:InteropVar) device(-4)

  //expected-error@+1 {{directive '#pragma omp interop' cannot contain more than one 'nowait' clause}}
  #pragma omp interop nowait init(target:InteropVar) nowait
}
#endif
#ifdef WITHOUTDEF
void foo() {
  int InteropVar;
  //expected-error@+1 {{'omp_interop_t' type not found; include <omp.h>}}
  #pragma omp interop init(prefer_type(1,"sycl",3),target:InteropVar) nowait
  //expected-error@+1 {{'omp_interop_t' type not found; include <omp.h>}}
  #pragma omp interop use(InteropVar) nowait
}
#endif
