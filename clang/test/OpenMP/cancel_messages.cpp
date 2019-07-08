// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
#pragma omp cancellation       // expected-error {{expected an OpenMP directive}}
#pragma omp cancel // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  ;
#pragma omp parallel
  {
#pragma omp cancel // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancel parallel untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp cancel'}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancel unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
#pragma omp parallel
  {
#pragma omp cancel unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancel sections(       // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancel for, )          // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancel taskgroup()     // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancel parallel, if    // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  if (argc)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    if (argc) {
#pragma omp taskgroup
#pragma omp task
#pragma omp parallel
      {
#pragma omp cancel taskgroup // expected-error {{region cannot be closely nested inside 'parallel' region}}
      }
    }
#pragma omp parallel
#pragma omp taskgroup
  {
#pragma omp cancel taskgroup // expected-error {{region cannot be closely nested inside 'taskgroup' region}}
  }
#pragma omp parallel
  {
#pragma omp cancel for // expected-error {{region cannot be closely nested inside 'parallel' region}}
  }
#pragma omp task
  {
#pragma omp cancel sections // expected-error {{region cannot be closely nested inside 'task' region}}
  }
#pragma omp sections
  {
#pragma omp cancel parallel allocate(argc) // expected-error {{region cannot be closely nested inside 'sections' region}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp cancel'}}
  }
  while (argc)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    while (argc) {
#pragma omp cancel sections // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    }
  do
#pragma omp cancel parallel // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    while (argc)
      ;
  do {
#pragma omp cancel taskgroup // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  } while (argc);
  switch (argc)
#pragma omp cancel parallel // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    switch (argc)
    case 1:
#pragma omp cancel sections // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  switch (argc)
  case 1: {
#pragma omp cancel for // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  }
  switch (argc) {
#pragma omp cancel taskgroup // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  case 1:
#pragma omp cancel parallel // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    break;
  default: {
#pragma omp cancel sections // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  } break;
  }
  for (;;)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    for (;;) {
#pragma omp cancel taskgroup // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    }
label:
#pragma omp cancel parallel // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
label1 : {
#pragma omp cancel sections // expected-error {{orphaned 'omp cancel' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
}

  return 0;
}

