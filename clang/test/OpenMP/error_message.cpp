// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

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
#pragma omp error // expected-error {{ERROR}}
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
#pragma omp error // expected-error {{ERROR}}
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
// expected-error@+1 {{ERROR}}
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
// expected-error@+1 {{ERROR}}
#pragma omp error
label1 : {
// expected-error@+1 {{ERROR}}
#pragma omp error
}
if (1)
  label2:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}

  return tmain(argc);
}
