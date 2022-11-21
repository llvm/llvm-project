// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

template <class T>
T tmain(T argc) {
  if (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    if (argc) {
#pragma omp error
    }
  while (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc) {
#pragma omp error
    }
  do
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp error
  } while (argc);
  switch (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp error
  }
  switch (argc) {
#pragma omp error
  case 1:
#pragma omp error
    break;
  default: {
#pragma omp error
  } break;
  }
  for (;;)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    for (;;) {
#pragma omp error
    }
label:
#pragma omp error
label1 : {
#pragma omp error
}
if (1)
  label2:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}

  return T();
}

int main(int argc, char **argv) {
#pragma omp error
  ;
#pragma omp error untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp error'}}
#pragma omp error unknown // expected-warning {{extra tokens at the end of '#pragma omp error' are ignored}}
  if (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    if (argc) {
#pragma omp error
    }
  while (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc) {
#pragma omp error
    }
  do
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp error
  } while (argc);
  switch (argc)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp error
  }
  switch (argc) {
#pragma omp error
  case 1:
#pragma omp error
    break;
  default: {
#pragma omp error
  } break;
  }
  for (;;)
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}
    for (;;) {
#pragma omp error
    }
label:
#pragma omp error
label1 : {
#pragma omp error
}
if (1)
  label2:
#pragma omp error // expected-error {{'#pragma omp error' cannot be an immediate substatement}}

  return tmain(argc);
}
