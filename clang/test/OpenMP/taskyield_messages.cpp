// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

template <class T>
T tmain(T argc) {
#pragma omp taskyield allocate(argc) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp taskyield'}}
  ;
#pragma omp taskyield untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp taskyield'}}
#pragma omp taskyield unknown // expected-warning {{extra tokens at the end of '#pragma omp taskyield' are ignored}}
  if (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    if (argc) {
#pragma omp taskyield
    }
  while (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    while (argc) {
#pragma omp taskyield
    }
  do
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp taskyield
  } while (argc);
  switch (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp taskyield
  }
  switch (argc) {
#pragma omp taskyield
  case 1:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    break;
  default: {
#pragma omp taskyield
  } break;
  }
  for (;;)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    for (;;) {
#pragma omp taskyield
    }
label:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
label1 : {
#pragma omp taskyield
}
if (1)
  label2:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}

  return T();
}

int main(int argc, char **argv) {
#pragma omp taskyield
  ;
#pragma omp taskyield untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp taskyield'}}
#pragma omp taskyield unknown // expected-warning {{extra tokens at the end of '#pragma omp taskyield' are ignored}}
  if (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    if (argc) {
#pragma omp taskyield
    }
  while (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    while (argc) {
#pragma omp taskyield
    }
  do
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp taskyield
  } while (argc);
  switch (argc)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp taskyield
  }
  switch (argc) {
#pragma omp taskyield
  case 1:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    break;
  default: {
#pragma omp taskyield
  } break;
  }
  for (;;)
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
    for (;;) {
#pragma omp taskyield
    }
label:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}
label1 : {
#pragma omp taskyield
}
if (1)
  label2:
#pragma omp taskyield // expected-error {{'#pragma omp taskyield' cannot be an immediate substatement}}

  return tmain(argc);
}
