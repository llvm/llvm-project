// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

template <class T>
T tmain(T argc) {
#pragma omp barrier
  ;
#pragma omp barrier allocate(argc)  // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp barrier'}}
#pragma omp barrier untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp barrier'}}
#pragma omp barrier unknown // expected-warning {{extra tokens at the end of '#pragma omp barrier' are ignored}}
  if (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    if (argc) {
#pragma omp barrier
    }
  while (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    while (argc) {
#pragma omp barrier
    }
  do
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp barrier
  } while (argc);
  switch (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp barrier
  }
  switch (argc) {
#pragma omp barrier
  case 1:
#pragma omp barrier
    break;
  default: {
#pragma omp barrier
  } break;
  }
  for (;;)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    for (;;) {
#pragma omp barrier
    }
label:
#pragma omp barrier
label1 : {
#pragma omp barrier
}

  return T();
}

int main(int argc, char **argv) {
#pragma omp barrier
  ;
#pragma omp barrier untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp barrier'}}
#pragma omp barrier unknown // expected-warning {{extra tokens at the end of '#pragma omp barrier' are ignored}}
  if (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    if (argc) {
#pragma omp barrier
    }
  while (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    while (argc) {
#pragma omp barrier
    }
  do
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp barrier
  } while (argc);
  switch (argc)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp barrier
  }
  switch (argc) {
#pragma omp barrier
  case 1:
#pragma omp barrier
    break;
  default: {
#pragma omp barrier
  } break;
  }
  for (;;)
#pragma omp barrier // expected-error {{'#pragma omp barrier' cannot be an immediate substatement}}
    for (;;) {
#pragma omp barrier
    }
label:
#pragma omp barrier
label1 : {
#pragma omp barrier
}

  return tmain(argc);
}
