// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

template <class T>
T tmain(T argc) {
#pragma omp taskwait
  ;
#pragma omp taskwait untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp taskwait'}}
#pragma omp taskwait unknown // expected-warning {{extra tokens at the end of '#pragma omp taskwait' are ignored}}
  if (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    if (argc) {
#pragma omp taskwait
    }
  while (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    while (argc) {
#pragma omp taskwait
    }
  do
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp taskwait
  } while (argc);
  switch (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp taskwait
  }
  switch (argc) {
#pragma omp taskwait
  case 1:
#pragma omp taskwait
    break;
  default: {
#pragma omp taskwait
  } break;
  }
  for (;;)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    for (;;) {
#pragma omp taskwait
    }
label:
#pragma omp taskwait
label1 : {
#pragma omp taskwait
}

  return T();
}

int main(int argc, char **argv) {
#pragma omp taskwait
  ;
#pragma omp taskwait untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp taskwait'}}
#pragma omp taskwait unknown // expected-warning {{extra tokens at the end of '#pragma omp taskwait' are ignored}}
  if (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    if (argc) {
#pragma omp taskwait
    }
  while (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    while (argc) {
#pragma omp taskwait
    }
  do
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp taskwait
  } while (argc);
  switch (argc)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp taskwait
  }
  switch (argc) {
#pragma omp taskwait
  case 1:
#pragma omp taskwait
    break;
  default: {
#pragma omp taskwait
  } break;
  }
  for (;;)
#pragma omp taskwait // expected-error {{'#pragma omp taskwait' cannot be an immediate substatement}}
    for (;;) {
#pragma omp taskwait
    }
label:
#pragma omp taskwait
label1 : {
#pragma omp taskwait
}

  return tmain(argc);
}
