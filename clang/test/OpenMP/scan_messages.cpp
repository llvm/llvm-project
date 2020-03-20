// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 %s

template <class T>
T tmain(T argc) {
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp scan
    ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp scan allocate(argc)  // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp scan'}}
#pragma omp scan untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp scan'}}
#pragma omp scan unknown // expected-warning {{extra tokens at the end of '#pragma omp scan' are ignored}}
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    if (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    if (argc) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc)
      ;
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan
  } while (argc);
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp scan
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan
  case 1:
#pragma omp scan
    break;
  default: {
#pragma omp scan
  } break;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    for (;;) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label1 : {
#pragma omp scan
}}

  return T();
}

int main(int argc, char **argv) {
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan
  ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp scan'}}
#pragma omp scan unknown // expected-warning {{extra tokens at the end of '#pragma omp scan' are ignored}}
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  if (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    if (argc) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc)
      ;
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan
  } while (argc);
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
  switch (argc)
  case 1: {
#pragma omp scan
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan
  case 1:
#pragma omp scan
    break;
  default: {
#pragma omp scan
  } break;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    for (;;) {
#pragma omp scan
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label1 : {
#pragma omp scan
}
}

  return tmain(argc);
}
