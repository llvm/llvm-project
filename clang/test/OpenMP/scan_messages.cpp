// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 %s

template <class T>
T tmain(T argc) {
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp scan // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
    ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp scan allocate(argc)  // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp scan'}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
#pragma omp scan untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp scan'}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
#pragma omp scan unknown // expected-warning {{extra tokens at the end of '#pragma omp scan' are ignored}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    if (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    if (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc)
      ;
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan inclusive(argc)
  } while (argc);
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  switch (argc)
  case 1: {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan exclusive(argc) // expected-note 2 {{previous 'scan' directive used here}}
  case 1:
#pragma omp scan exclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
    break;
  default: {
#pragma omp scan exclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
  } break;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    for (;;) {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan exclusive(argc)
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label1 : {
#pragma omp scan inclusive(argc)
}}

  return T();
}

int main(int argc, char **argv) {
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan inclusive(argc) inclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan exclusive(argc) inclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan exclusive(argc) exclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scan untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp scan'}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
#pragma omp scan unknown // expected-warning {{extra tokens at the end of '#pragma omp scan' are ignored}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  if (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    if (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    while (argc)
      ;
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan exclusive(argc)
  } while (argc);
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  switch (argc)
  case 1: {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan inclusive(argc) // expected-note 2 {{previous 'scan' directive used here}}
  case 1:
#pragma omp scan inclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
    break;
  default: {
#pragma omp scan inclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
  } break;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}}
    for (;;) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan inclusive(argc)
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
label1 : {
#pragma omp scan inclusive(argc)
}
}

  return tmain(argc);
}
