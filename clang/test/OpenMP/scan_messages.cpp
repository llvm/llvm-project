// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 150 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 150 %s

template <class T>
T tmain() {
  static T argc;
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
#pragma omp for simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
    if (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    if (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    while (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    while (argc)
      ;
#pragma omp simd reduction(inscan, +: argc) // expected-error {{the inscan reduction list item must appear as a list item in an 'inclusive' or 'exclusive' clause on an inner 'omp scan' directive}}
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  } while (argc);
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    switch (argc)
    case 1:
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  switch (argc)
  case 1: {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  }
#pragma omp simd reduction(inscan, +: argc) // expected-error {{the inscan reduction list item must appear as a list item in an 'inclusive' or 'exclusive' clause on an inner 'omp scan' directive}}
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  case 1:
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    break;
  default: {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  } break;
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    for (;;) {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan exclusive(argc)
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan inclusive(argc) // expected-note {{previous 'scan' directive used here}}
#pragma omp scan inclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
label1 : {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
}}

  return T();
}

int main() {
  static int argc;
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan inclusive(argc) inclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan exclusive(argc) inclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan exclusive(argc) exclusive(argc) // expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  ;
  }
#pragma omp simd reduction(inscan, +: argc) // expected-error {{the inscan reduction list item must appear as a list item in an 'inclusive' or 'exclusive' clause on an inner 'omp scan' directive}}
  for (int i = 0; i < 10; ++i) {
#pragma omp scan untied  // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp scan'}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
#pragma omp scan unknown // expected-warning {{extra tokens at the end of '#pragma omp scan' are ignored}} expected-error {{exactly one of 'inclusive' or 'exclusive' clauses is expected}}
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  if (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    if (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}} expected-error {{the list item must appear in 'reduction' clause with the 'inscan' modifier of the parent directive}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  while (argc)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    while (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}} expected-error {{the list item must appear in 'reduction' clause with the 'inscan' modifier of the parent directive}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  do
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    while (argc)
      ;
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  do {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  } while (argc);
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  switch (argc)
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    switch (argc)
    case 1:
#pragma omp scan exclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}} expected-error {{the list item must appear in 'reduction' clause with the 'inscan' modifier of the parent directive}}
  switch (argc)
  case 1: {
#pragma omp scan exclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}} expected-error {{the list item must appear in 'reduction' clause with the 'inscan' modifier of the parent directive}}
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  switch (argc) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  case 1:
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    break;
  default: {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
  } break;
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i)
  for (;;)
#pragma omp scan inclusive(argc) // expected-error {{'#pragma omp scan' cannot be an immediate substatement}} expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
    for (;;) {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}} expected-error {{the list item must appear in 'reduction' clause with the 'inscan' modifier of the parent directive}}
    }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
label:
#pragma omp scan inclusive(argc)
  }
#pragma omp simd reduction(inscan, +: argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan inclusive(argc) // expected-note {{previous 'scan' directive used here}}
#pragma omp scan inclusive(argc) // expected-error {{exactly one 'scan' directive must appear in the loop body of an enclosing directive}}
label1 : {
#pragma omp scan inclusive(argc) // expected-error {{orphaned 'omp scan' directives are prohibited; perhaps you forget to enclose the directive into a for, simd, for simd, parallel for, or parallel for simd region?}}
}
}

  return tmain<int>(); // expected-note {{in instantiation of function template specialization 'tmain<int>' requested here}}
}
