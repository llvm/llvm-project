// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp    \
// RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify -fopenmp \
// RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

int disp_variant();
#pragma omp declare variant(disp_variant) \
    match(construct = {dispatch}, device = {arch(arm)})
int disp_call();

struct Obj {
  int disp_method_variant1();
  #pragma omp declare variant(disp_method_variant1)                            \
    match(construct={dispatch}, device={arch(arm)})
  int disp_method1();
  int disp_method_variant2();
  #pragma omp declare variant(disp_method_variant2)                            \
    match(construct={dispatch}, device={arch(arm)})
  int disp_method2();
};

void testit_one(int dnum) {
  // expected-error@+1 {{cannot contain more than one 'device' clause}}
  #pragma omp dispatch device(dnum) device(3)
  disp_call();

  // expected-error@+1 {{cannot contain more than one 'nowait' clause}}
  #pragma omp dispatch nowait device(dnum) nowait
  disp_call();

  // expected-error@+1 {{expected '(' after 'novariants'}}
  #pragma omp dispatch novariants
  disp_call();

  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp dispatch novariants (
  disp_call();

  // expected-error@+1 {{cannot contain more than one 'novariants' clause}}
  #pragma omp dispatch novariants(dnum> 4) novariants(3)
  disp_call();

  // expected-error@+1 {{use of undeclared identifier 'x'}}
  #pragma omp dispatch novariants(x)
  disp_call();
  
  // expected-error@+1 {{expected '(' after 'nocontext'}}
  #pragma omp dispatch nocontext
  disp_call();

  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp dispatch nocontext (
  disp_call();

  // expected-error@+1 {{cannot contain more than one 'nocontext' clause}}
  #pragma omp dispatch nocontext(dnum> 4) nocontext(3)
  disp_call();

  // expected-error@+1 {{use of undeclared identifier 'x'}}
  #pragma omp dispatch nocontext(x)
  disp_call();
}

void testit_two() {
  //expected-error@+2 {{cannot return from OpenMP region}}
  #pragma omp dispatch
  return disp_call();
}

void testit_three(int (*fptr)(void), Obj *obj, int (Obj::*mptr)(void)) {
  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  fptr();

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  (obj->*mptr)();

  int ret;

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  ret = fptr();

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  ret = (obj->*mptr)();
}

void testit_four(int *x, int y, Obj *obj)
{
  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  *x = y;

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  y = disp_call() + disp_call();

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  y = (y = disp_call());

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  y += disp_call();

  //expected-error@+2 {{statement after '#pragma omp dispatch' must be a direct call to a target function or an assignment to one}}
  #pragma omp dispatch
  for (int I = 0; I < 8; ++I) {
    disp_call();
  }
}
