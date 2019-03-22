// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

struct S1 { // expected-note 2 {{declared here}}
  int a;
};

template <class T>
T tmain(T argc) {
#pragma omp flush
  ;
#pragma omp flush untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp flush'}}
#pragma omp flush unknown // expected-warning {{extra tokens at the end of '#pragma omp flush' are ignored}}
  if (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    if (argc) {
#pragma omp flush
    }
  while (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    while (argc) {
#pragma omp flush
    }
  do
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp flush
  } while (argc);
  switch (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp flush
  }
  switch (argc) {
#pragma omp flush
  case 1:
#pragma omp flush
    break;
  default: {
#pragma omp flush
  } break;
  }
  for (;;)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    for (;;) {
#pragma omp flush
    }
label:
#pragma omp flush
label1 : {
#pragma omp flush
}

#pragma omp flush
#pragma omp flush(                              // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush()                             // expected-error {{expected expression}}
#pragma omp flush(argc                          // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush(argc,                         // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush(argc)
#pragma omp flush(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp flush(argc) flush(argc) // expected-warning {{extra tokens at the end of '#pragma omp flush' are ignored}}
#pragma omp parallel flush(argc) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  ;
  return T();
}

int main(int argc, char **argv) {
#pragma omp flush
  ;
#pragma omp flush untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp flush'}}
#pragma omp flush unknown // expected-warning {{extra tokens at the end of '#pragma omp flush' are ignored}}
  if (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    if (argc) {
#pragma omp flush
    }
  while (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    while (argc) {
#pragma omp flush
    }
  do
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp flush
  } while (argc);
  switch (argc)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp flush
  }
  switch (argc) {
#pragma omp flush
  case 1:
#pragma omp flush
    break;
  default: {
#pragma omp flush
  } break;
  }
  for (;;)
#pragma omp flush // expected-error {{'#pragma omp flush' cannot be an immediate substatement}}
    for (;;) {
#pragma omp flush
    }
label:
#pragma omp flush
label1 : {
#pragma omp flush
}

#pragma omp flush
#pragma omp flush(                              // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush()                             // expected-error {{expected expression}}
#pragma omp flush(argc                          // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush(argc,                         // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush(argc)
#pragma omp flush(S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp flush(argc) flush(argc) // expected-warning {{extra tokens at the end of '#pragma omp flush' are ignored}}
#pragma omp parallel flush(argc) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  ;
  return tmain(argc);
}
