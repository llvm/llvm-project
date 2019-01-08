// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

int main(int argc, char **argv) {
#pragma omp cancellation       // expected-error {{expected an OpenMP directive}}
#pragma omp cancellation point // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  ;
#pragma omp parallel
  {
#pragma omp cancellation point // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancellation point parallel untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp cancellation point'}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancellation point unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
#pragma omp parallel
  {
#pragma omp cancellation point unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancellation point sections(       // expected-warning {{extra tokens at the end of '#pragma omp cancellation point' are ignored}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancellation point for, )          // expected-warning {{extra tokens at the end of '#pragma omp cancellation point' are ignored}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancellation point taskgroup()     // expected-warning {{extra tokens at the end of '#pragma omp cancellation point' are ignored}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
#pragma omp cancellation point parallel, if    // expected-warning {{extra tokens at the end of '#pragma omp cancellation point' are ignored}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  if (argc)
#pragma omp cancellation point for // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    if (argc) {
#pragma omp taskgroup
#pragma omp task
#pragma omp parallel
      {
#pragma omp cancellation point taskgroup // expected-error {{region cannot be closely nested inside 'parallel' region}}
      }
    }
#pragma omp parallel
#pragma omp taskgroup
  {
#pragma omp cancellation point taskgroup // expected-error {{region cannot be closely nested inside 'taskgroup' region}}
  }
#pragma omp parallel
  {
#pragma omp cancellation point for // expected-error {{region cannot be closely nested inside 'parallel' region}}
  }
#pragma omp task
  {
#pragma omp cancellation point sections // expected-error {{region cannot be closely nested inside 'task' region}}
  }
#pragma omp sections
  {
#pragma omp cancellation point parallel // expected-error {{region cannot be closely nested inside 'sections' region}}
  }
  while (argc)
#pragma omp cancellation point for // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    while (argc) {
#pragma omp cancellation point sections // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    }
  do
#pragma omp cancellation point parallel // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    while (argc)
      ;
  do {
#pragma omp cancellation point taskgroup // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  } while (argc);
  switch (argc)
#pragma omp cancellation point parallel // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    switch (argc)
    case 1:
#pragma omp cancellation point sections // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  switch (argc)
  case 1: {
#pragma omp cancellation point for // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  }
  switch (argc) {
#pragma omp cancellation point taskgroup // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  case 1:
#pragma omp cancellation point parallel // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    break;
  default: {
#pragma omp cancellation point sections // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
  } break;
  }
  for (;;)
#pragma omp cancellation point for // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    for (;;) {
#pragma omp cancellation point taskgroup // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
    }
label:
#pragma omp cancellation point parallel // expected-error {{'#pragma omp cancellation point' cannot be an immediate substatement}} expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
label1 : {
#pragma omp cancellation point sections // expected-error {{orphaned 'omp cancellation point' directives are prohibited; perhaps you forget to enclose the directive into a region?}}
}

  return 0;
}

