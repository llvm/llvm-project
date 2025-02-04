// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

template <class T>
T tmain(T argc) {
  if (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    if (argc) {
#pragma omp error // expected-error {{ERROR}}
    }
  while (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc) {
#pragma omp error // expected-error {{ERROR}}
    }
  do
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp error // expected-error {{ERROR}}
  } while (argc);
  switch (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp error // expected-error {{ERROR}}
  }
  switch (argc) {
#pragma omp error // expected-error {{ERROR}}
  case 1:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    break;
  default: {
#pragma omp error // expected-error {{ERROR}}
  } break;
  }
  for (;;)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    for (;;) {
#pragma omp error // expected-error {{ERROR}}
    }
label:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
label1 : {
#pragma omp error // expected-error {{ERROR}}
}
if (1)
  label2:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}

// expected-error@+1 {{ERROR}}
#pragma omp error at() // expected-error {{expected 'compilation' or 'execution' in OpenMP clause 'at'}}

// expected-error@+1 {{ERROR}}
#pragma omp error at(up) // expected-error {{expected 'compilation' or 'execution' in OpenMP clause 'at'}}

// expected-error@+3 {{ERROR}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp error at(up(a)) // expected-error {{expected 'compilation' or 'execution' in OpenMP clause 'at'}}

#pragma omp error at(execution) // no error

#pragma omp error at(compilation) // expected-error {{ERROR}}

// expected-error@+1 {{ERROR}}
#pragma omp error at(compilation) at(execution) //expected-error {{directive '#pragma omp error' cannot contain more than one 'at' clause}}

// expected-error@+1 {{ERROR}}
#pragma omp error severity() // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}
// expected-error@+1 {{ERROR}}
#pragma omp error severity(xyz) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}
// expected-error@+1 {{ERROR}}
#pragma omp error severity(fatal) severity(warning) // expected-error {{directive '#pragma omp error' cannot contain more than one 'severity' clause}}

// expected-error@+1 {{ERROR}}
#pragma omp error severity(fatal) severity(fatal) // expected-error {{directive '#pragma omp error' cannot contain more than one 'severity' clause}}

// expected-warning@+2 {{WARNING}}
// expected-warning@+1 {{WARNING}}
#pragma omp error severity(warning) severity(warning) // expected-error {{directive '#pragma omp error' cannot contain more than one 'severity' clause}}

// expected-warning@+1 {{WARNING}}
#pragma omp error severity(warning) // expected-warning {{WARNING}}
#pragma omp error severity(fatal) // expected-error {{ERROR}}

// expected-warning@+1 {{WARNING}}
#pragma omp error at(compilation) severity(warning) // expected-warning {{WARNING}}
#pragma omp error at(execution) severity(warning) // no error, diagnosic at runtime
#pragma omp error at(compilation) severity(fatal) // expected-error {{ERROR}}
#pragma omp error at(execution) severity(fatal) // no error, error at runtime

#pragma omp error message("GPU compiler is needed.") // expected-error {{GPU compiler is needed}}
#pragma omp error at(compilation) message("GPU compiler is needed.") // expected-error {{GPU compiler is needed}}
#pragma omp error at(execution) message("GPU compiler is needed.") // no error
// expected-warning@+1 {{GPU compiler is needed.}}
#pragma omp error severity(warning) message("GPU compiler is needed.") // expected-warning {{GPU compiler is needed.}}
#pragma omp error severity(fatal) message("GPU compiler is needed.") // expected-error {{GPU compiler is needed}}
// expected-warning@+1 {{GPU compiler is needed.}}
#pragma omp error at(compilation) severity(warning) message("GPU compiler is needed.") // expected-warning {{GPU compiler is needed.}}
#pragma omp error at(compilation) severity(fatal) message("GPU compiler is needed.") // expected-error {{GPU compiler is needed.}}
#pragma omp error at(execution) severity(warning) message("GPU compiler is needed.") // no warning warning will emit at runtime.
#pragma omp error at(execution) severity(fatal) message("GPU compiler is needed.") // no warning warning will emit at runtime.

// expected-error@+1 {{GPU compiler is needed.}}
#pragma omp error message("GPU compiler is needed.") message("GPU compiler is needed.") // expected-error {{directive '#pragma omp error' cannot contain more than one 'message' clause}}
  int a;
// expected-warning@+1 {{expected string literal in 'clause message' - ignoring}}
#pragma omp error message(a) // expected-error {{ERROR}}
// expected-error@+1 {{ERROR}}
#pragma omp error message() // expected-error {{expected expression}}
  return T();
}

#pragma omp error at(execution) // expected-error {{unexpected 'execution' modifier in non-executable context}}

#pragma omp error at(compilation) // expected-error {{ERROR}}
class A {

#pragma omp error at(compilation) // expected-error {{ERROR}}

#pragma omp error at(execution) // expected-error {{unexpected 'execution' modifier in non-executable context}}
  int A;
};

int main(int argc, char **argv) {
// expected-error@+1 {{ERROR}}
#pragma omp error
  ;
// expected-error@+1 {{ERROR}}
#pragma omp error untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp error'}}
  if (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    if (argc) {
// expected-error@+1 {{ERROR}}
#pragma omp error
    }
  while (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc) {
// expected-error@+1 {{ERROR}}
#pragma omp error
    }
  do
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
// expected-error@+1 {{ERROR}}
#pragma omp error
  } while (argc);
  switch (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
// expected-error@+1 {{ERROR}}
#pragma omp error
  }
  switch (argc) {
// expected-error@+1 {{ERROR}}
#pragma omp error
  case 1:
// expected-error@+1 {{'#pragma omp error' cannot be an immediate substatement}}
#pragma omp error
    break;
  default: {
// expected-error@+1 {{ERROR}}
#pragma omp error
  } break;
  }
  for (;;)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    for (;;) {
// expected-error@+1 {{ERROR}}
#pragma omp error
    }
label:
// expected-error@+1 {{'#pragma omp error' cannot be an immediate substatement}}
#pragma omp error
label1 : {
// expected-error@+1 {{ERROR}}
#pragma omp error
}
if (1)
  label2:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}

  return tmain(argc);// expected-note {{in instantiation of function template specialization 'tmain<int>' requested here}}
}
