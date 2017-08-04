// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

int main(int argc, char **argv) {
#pragma omp cancellation       // expected-error {{expected an OpenMP directive}}
#pragma omp cancel // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  ;
#pragma omp parallel
  {
#pragma omp cancel // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancel parallel untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp cancel'}}
#pragma omp cancel unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
#pragma omp parallel
  {
#pragma omp cancel unknown         // expected-error {{one of 'for', 'parallel', 'sections' or 'taskgroup' is expected}}
  }
#pragma omp cancel sections(       // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
#pragma omp cancel for, )          // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
#pragma omp cancel taskgroup()     // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
#pragma omp cancel parallel, if    // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
  if (argc)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
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
#pragma omp cancel parallel // expected-error {{region cannot be closely nested inside 'sections' region}}
  }
  while (argc)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
    while (argc) {
#pragma omp cancel sections
    }
  do
#pragma omp cancel parallel // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
    while (argc)
      ;
  do {
#pragma omp cancel taskgroup
  } while (argc);
  switch (argc)
#pragma omp cancel parallel // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
    switch (argc)
    case 1:
#pragma omp cancel sections
  switch (argc)
  case 1: {
#pragma omp cancel for
  }
  switch (argc) {
#pragma omp cancel taskgroup
  case 1:
#pragma omp cancel parallel
    break;
  default: {
#pragma omp cancel sections
  } break;
  }
  for (;;)
#pragma omp cancel for // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
    for (;;) {
#pragma omp cancel taskgroup
    }
label:
#pragma omp cancel parallel // expected-error {{'#pragma omp cancel' cannot be an immediate substatement}}
label1 : {
#pragma omp cancel sections
}

  return 0;
}

