// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

struct S1 { // expected-note 2 {{declared here}}
  int a;
};

template <class T>
T tmain(T argc) {
#pragma omp flush allocate(argc) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp flush'}}
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
#pragma omp flush seq_cst // expected-error {{unexpected OpenMP clause 'seq_cst' in directive '#pragma omp flush'}}
#pragma omp flush acq_rel // omp45-error {{unexpected OpenMP clause 'acq_rel' in directive '#pragma omp flush'}}
#pragma omp flush acquire // omp45-error {{unexpected OpenMP clause 'acquire' in directive '#pragma omp flush'}}
#pragma omp flush release // omp45-error {{unexpected OpenMP clause 'release' in directive '#pragma omp flush'}}
#pragma omp flush relaxed // expected-error {{unexpected OpenMP clause 'relaxed' in directive '#pragma omp flush'}}
#pragma omp flush seq_cst // expected-error {{unexpected OpenMP clause 'seq_cst' in directive '#pragma omp flush'}}
#pragma omp flush acq_rel acquire // omp45-error {{unexpected OpenMP clause 'acq_rel' in directive '#pragma omp flush'}} omp45-error {{unexpected OpenMP clause 'acquire' in directive '#pragma omp flush'}} omp50-error {{directive '#pragma omp flush' cannot contain more than one 'acq_rel', 'acquire' or 'release' clause}} omp50-note {{'acq_rel' clause used here}}
#pragma omp flush release acquire // omp45-error {{unexpected OpenMP clause 'release' in directive '#pragma omp flush'}} omp45-error {{unexpected OpenMP clause 'acquire' in directive '#pragma omp flush'}} omp50-error {{directive '#pragma omp flush' cannot contain more than one 'acq_rel', 'acquire' or 'release' clause}} omp50-note {{'release' clause used here}}
#pragma omp flush acq_rel (argc) // omp45-error {{unexpected OpenMP clause 'acq_rel' in directive '#pragma omp flush'}} expected-warning {{extra tokens at the end of '#pragma omp flush' are ignored}}
#pragma omp flush(argc) acq_rel // omp45-error {{unexpected OpenMP clause 'acq_rel' in directive '#pragma omp flush'}} omp50-error {{'flush' directive with memory order clause 'acq_rel' cannot have the list}} omp50-note {{memory order clause 'acq_rel' is specified here}}
  return tmain(argc);
}
