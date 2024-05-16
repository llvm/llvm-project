// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 \
// RUN: -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp \
// RUN: -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp -fopenmp-version=52 \
// RUN: -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 \
// RUN: -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd \
// RUN: -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp52 -fopenmp-simd -fopenmp-version=52 \
// RUN: -ferror-limit 100 %s -Wuninitialized

struct S1 { // expected-note 2 {{declared here}}
  int a;
} s;

#pragma omp depobj(0) depend(in:s) // expected-error {{unexpected OpenMP directive '#pragma omp depobj'}}
void foo() {
#pragma omp depobj(0) depend(in:s) // expected-error {{'omp_depend_t' type not found; include <omp.h>}} expected-error {{expected lvalue expression}}}
}

typedef void *omp_depend_t;

template <class T>
T tmain(T argc) {
  omp_depend_t x;
#pragma omp depobj() allocate(argc) // expected-error {{expected expression}} expected-error {{expected depobj expression}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp depobj'}}
  ;
#pragma omp depobj(x) untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp depobj'}}
#pragma omp depobj(x) unknown // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
  if (argc)
#pragma omp depobj(x) destroy // omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    if (argc) {
#pragma omp depobj(x) depend(in:s)
    }
  while (argc)
#pragma omp depobj(x)update(inout) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    while (argc) {
#pragma omp depobj(x) depend(in:s)
    }
  do
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp depobj(x) depend(in:s)
  } while (argc);
  switch (argc)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp depobj(x) depend(in:s)
  }
  switch (argc) {
#pragma omp depobj(x) depend(in:s)
  case 1:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    break;
  default: {
#pragma omp depobj(x) depend(in:s)
  } break;
  }
  for (;;)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    for (;;) {
#pragma omp depobj(x) depend(in:s)
    }
label:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
label1 : {
#pragma omp depobj(x) depend(in:s)
}

#pragma omp depobj                               // expected-error {{expected depobj expression}}
#pragma omp depobj(                              // expected-error {{expected expression}} expected-error {{expected depobj expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp depobj()                             // expected-error {{expected expression}} expected-error {{expected depobj expression}}
#pragma omp depobj(argc                          // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}}
#pragma omp depobj(argc,                         // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp depobj(argc)                         // expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp depobj(S1) // expected-error {{'S1' does not refer to a value}} expected-error {{expected depobj expression}}
#pragma omp depobj(argc) depobj(argc) // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}}
#pragma omp parallel depobj(argc) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  ;
  return T();
}

int main(int argc, char **argv) {
omp_depend_t x;
#pragma omp depobj(x) depend(in:s)
  ;
#pragma omp depobj(x) untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp depobj'}}
#pragma omp depobj(x) unknown // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
  if (argc)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    if (argc) {
#pragma omp depobj(x) depend(in:s)
    }
  while (argc)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    while (argc) {
#pragma omp depobj(x) depend(in:s)
    }
  do
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp depobj(x) depend(in:s)
  } while (argc);
  switch (argc)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp depobj(x) depend(in:s)
  }
  switch (argc) {
#pragma omp depobj(x) depend(in:s)
  case 1:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    break;
  default: {
#pragma omp depobj(x) depend(in:s)
  } break;
  }
  for (;;)
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
    for (;;) {
#pragma omp depobj(x) depend(in:s)
    }
label:
#pragma omp depobj(x) depend(in:s) // expected-error {{'#pragma omp depobj' cannot be an immediate substatement}}
label1 : {
#pragma omp depobj(x) depend(in:s)
}

#pragma omp depobj                               // expected-error {{expected depobj expression}}
#pragma omp depobj(                              // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected depobj expression}}
#pragma omp depobj()                             // expected-error {{expected expression}} expected-error {{expected depobj expression}}
#pragma omp depobj(argc                          // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp depobj(argc,                         // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp depobj(argc)                         // expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp depobj(S1) // expected-error {{'S1' does not refer to a value}} expected-error {{expected depobj expression}}
#pragma omp depobj(argc) depobj(argc) // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}} expected-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
#pragma omp parallel depobj(argc) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  ;
#pragma omp depobj(x) seq_cst // expected-error {{unexpected OpenMP clause 'seq_cst' in directive '#pragma omp depobj'}}
#pragma omp depobj(x) depend(source: x) // omp52-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'depend'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}}
#pragma omp depobj(x) update // expected-error {{expected '(' after 'update'}}
#pragma omp depobj(x) update( // expected-error {{expected ')'}} expected-note {{to match this '('}} omp52-error {{expected 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'update'}} omp51-error {{expected 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'update'}} omp50-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'update'}}
#pragma omp depobj(x) update(sink // expected-error {{expected ')'}} expected-note {{to match this '('}} omp52-error {{expected 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'update'}} omp51-error {{expected 'in', 'out', 'inout', 'mutexinoutset' or 'inoutset' in OpenMP clause 'update'}} omp50-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'update'}}
#pragma omp depobj(x) destroy destroy //omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} omp50-error {{directive '#pragma omp depobj' cannot contain more than one 'destroy' clause}} // omp52-error {{directive '#pragma omp depobj' cannot contain more than one 'destroy' clause}} omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} omp51-error {{directive '#pragma omp depobj' cannot contain more than one 'destroy' clause}}
#pragma omp depobj(x) update(in) update(in) // expected-error {{directive '#pragma omp depobj' cannot contain more than one 'update' clause}}
#pragma omp depobj(x) depend(in: argc) destroy // omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} omp51-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}} omp50-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}}
#pragma omp depobj(x) destroy depend(in: argc) // omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} omp51-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}} omp50-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}}
#pragma omp depobj(x) depend(in: argc) update(mutexinoutset) // expected-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}}
#pragma omp depobj(x) update(inout) destroy // omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} omp51-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}} omp50-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}}
#pragma omp depobj(x) (x) depend(in: x) // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
#pragma omp depobj(x) (x) update(in) // expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
#pragma omp depobj(x) depend(in: argc) depend(out:argc) // expected-error {{exactly one of 'depend', 'destroy', or 'update' clauses is expected}}
#pragma omp depend(out:x) depobj(x) // expected-error {{expected an OpenMP directive}}
#pragma omp destroy depobj(x) // expected-error {{expected an OpenMP directive}}
#pragma omp update(out) depobj(x) // expected-error {{expected an OpenMP directive}}
#pragma omp depobj depend(in:x) (x) // expected-error {{expected depobj expression}} expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}} expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
#pragma omp depobj destroy (x) // omp52-error {{expected 'destroy' clause with an argument on '#pragma omp depobj' construct}} expected-error {{expected depobj expression}} expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
#pragma omp depobj update(in) (x) // expected-error {{expected depobj expression}} expected-warning {{extra tokens at the end of '#pragma omp depobj' are ignored}}
  return tmain(argc); // expected-note {{in instantiation of function template specialization 'tmain<int>' requested here}}
}
