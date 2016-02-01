// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s

void bar();

template <class T>
void foo() {
  T a = T();
// PARALLEL DIRECTIVE
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel region}}
  {
    bar();
  }
#pragma omp parallel
#pragma omp single
  bar();

#pragma omp parallel
#pragma omp master
  {
    bar();
  }
#pragma omp parallel
#pragma omp critical
  {
    bar();
  }
#pragma omp parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp task
  {
    bar();
  }
#pragma omp parallel
  {
#pragma omp taskyield
    bar();
  }
#pragma omp parallel
  {
#pragma omp barrier
    bar();
  }
#pragma omp parallel
  {
#pragma omp taskwait
    bar();
  }
#pragma omp parallel
  {
#pragma omp flush
    bar();
  }
#pragma omp parallel
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel
  {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SIMD DIRECTIVE
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// FOR DIRECTIVE
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a for region}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }

#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'for' region}}
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // OK
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// FOR SIMD DIRECTIVE
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// SECTIONS DIRECTIVE
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp master // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp master // OK
      {
        bar();
      }
    }
#pragma omp master // expected-error {{region cannot be closely nested inside 'sections' region}}
    bar();
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp critical(A) // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp critical // OK
      {
        bar();
      }
    }
#pragma omp critical(A) // expected-error {{statement in 'omp sections' directive must be enclosed into a section region}}
    bar();
  }
#pragma omp sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp taskyield
  }
#pragma omp sections
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'sections' region}}
  }
#pragma omp sections
  {
#pragma omp taskwait
  }
#pragma omp sections
  {
#pragma omp flush
  }
#pragma omp sections
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp sections
  {
#pragma omp atomic
    ++a;
  }
#pragma omp sections
  {
#pragma omp target
    ++a;
  }
#pragma omp sections
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp sections
  {
#pragma omp target enter data map(to: a)
  }
#pragma omp sections
  {
#pragma omp target exit data map(from: a)
  }
#pragma omp sections
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp sections
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp sections
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SECTION DIRECTIVE
#pragma omp section // expected-error {{orphaned 'omp section' directives are prohibited, it must be closely nested to a sections region}}
  {
    bar();
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp for // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp simd
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a section region}}
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
#pragma omp master // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
#pragma omp critical
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel
      {
#pragma omp single // OK
        {
          bar();
        }
#pragma omp for // OK
        for (int i = 0; i < 10; ++i)
          ;
#pragma omp for simd // OK
        for (int i = 0; i < 10; ++i)
          ;
#pragma omp sections // OK
        {
          bar();
        }
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel for
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel for simd
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel sections
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp task
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp taskyield
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp taskwait
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp flush
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp atomic
    ++a;
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp target
    ++a;
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp target parallel
    ++a;
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target enter data map(to: a)
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target exit data map(from: a)
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp teams // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SINGLE DIRECTIVE
#pragma omp single
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp master // expected-error {{region cannot be closely nested inside 'single' region}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp single
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp taskyield
    bar();
  }
#pragma omp single
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'single' region}}
    bar();
  }
#pragma omp single
  {
#pragma omp taskwait
    bar();
  }
#pragma omp single
  {
#pragma omp flush
    bar();
  }
#pragma omp single
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp single
  {
#pragma omp atomic
    ++a;
  }
#pragma omp single
  {
#pragma omp target
    ++a;
  }
#pragma omp single
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp single
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp single
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp single
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp single
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp single
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// MASTER DIRECTIVE
#pragma omp master
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp master // OK, though second 'master' is redundant
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp parallel
    {
#pragma omp master // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp master
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp taskyield
    bar();
  }
#pragma omp master
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'master' region}}
    bar();
  }
#pragma omp master
  {
#pragma omp taskwait
    bar();
  }
#pragma omp master
  {
#pragma omp flush
    bar();
  }
#pragma omp master
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp master
  {
#pragma omp atomic
    ++a;
  }
#pragma omp master
  {
#pragma omp target
    ++a;
  }
#pragma omp master
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp master
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp master
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp master
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp master
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp master
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// CRITICAL DIRECTIVE
#pragma omp critical
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp master // OK, though second 'master' is redundant
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp parallel
    {
#pragma omp master // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp critical
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp taskyield
    bar();
  }
#pragma omp critical
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'critical' region}}
    bar();
  }
#pragma omp critical
  {
#pragma omp taskwait
    bar();
  }
#pragma omp critical(Tuzik)
  {
#pragma omp critical(grelka)
    bar();
  }
#pragma omp critical(Belka) // expected-note {{previous 'critical' region starts here}}
  {
#pragma omp critical(Belka) // expected-error {{cannot nest 'critical' regions having the same name 'Belka'}}
    {
#pragma omp critical(Tuzik)
      {
#pragma omp parallel
#pragma omp critical(grelka)
        {
          bar();
        }
      }
    }
  }
#pragma omp critical
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp critical
  {
#pragma omp atomic
    ++a;
  }
#pragma omp critical
  {
#pragma omp target
    ++a;
  }
#pragma omp critical
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp critical
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp critical
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp critical
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp critical
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp critical
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// PARALLEL FOR DIRECTIVE
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel for region}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{region cannot be closely nested inside 'parallel for' region}}
    {
      bar();
    }
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'parallel for' region}}
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // OK
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// PARALLEL FOR SIMD DIRECTIVE
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd// expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
#pragma omp single
      {
        bar();
      }
#pragma omp for
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections
      {
        bar();
      }
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd// expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// PARALLEL SECTIONS DIRECTIVE
#pragma omp parallel sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp master // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp critical
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp taskyield
  }
#pragma omp parallel sections
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'parallel sections' region}}
  }
#pragma omp parallel sections
  {
#pragma omp taskwait
  }
#pragma omp parallel sections
  {
#pragma omp flush
  }
#pragma omp parallel sections
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel sections
  {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target enter data map(to: a)
  }
#pragma omp parallel sections
  {
#pragma omp target exit data map(from: a)
  }
#pragma omp parallel sections
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// TASK DIRECTIVE
#pragma omp task
#pragma omp for // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp sections // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp task
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a task region}}
  {
    bar();
  }
#pragma omp task
#pragma omp single // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
  bar();
#pragma omp task
#pragma omp master // expected-error {{region cannot be closely nested inside 'task' region}}
  bar();
#pragma omp task
#pragma omp critical
  bar();

#pragma omp task
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp task
#pragma omp task
  {
    bar();
  }
#pragma omp task
  {
#pragma omp taskyield
    bar();
  }
#pragma omp task
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'task' region}}
    bar();
  }
#pragma omp task
  {
#pragma omp taskwait
    bar();
  }
#pragma omp task
  {
#pragma omp flush
    bar();
  }
#pragma omp task
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp task
  {
#pragma omp atomic
    ++a;
  }
#pragma omp task
  {
#pragma omp target
    ++a;
  }
#pragma omp task
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp task
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp task
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp task
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp task
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp task
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// ORDERED DIRECTIVE
#pragma omp ordered
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp master // OK, though second 'ordered' is redundant
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp parallel for ordered
    for (int j = 0; j < 10; ++j) {
#pragma omp ordered // OK
      {
        bar();
      }
    }
  }
#pragma omp ordered
  {
#pragma omp parallel for simd ordered
    for (int j = 0; j < 10; ++j) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
      {
        bar();
      }
    }
  }
#pragma omp ordered
  {
#pragma omp parallel for simd ordered
    for (int j = 0; j < 10; ++j) {
#pragma omp ordered simd
      {
        bar();
      }
    }
  }
#pragma omp ordered
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp ordered
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp ordered
  {
#pragma omp taskyield
    bar();
  }
#pragma omp ordered
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'ordered' region}}
    bar();
  }
#pragma omp ordered
  {
#pragma omp taskwait
    bar();
  }
#pragma omp ordered
  {
#pragma omp flush
    bar();
  }
#pragma omp ordered
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp ordered
  {
#pragma omp atomic
    ++a;
  }
#pragma omp ordered
  {
#pragma omp target
    ++a;
  }
#pragma omp ordered
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp ordered
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp ordered
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp ordered
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp ordered
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp ordered
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'ordered' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// ATOMIC DIRECTIVE
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// TARGET DIRECTIVE
#pragma omp target
#pragma omp parallel
  bar();
#pragma omp target
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp sections
  {
    bar();
  }
#pragma omp target
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a target region}}
  {
    bar();
  }
#pragma omp target
#pragma omp single
  bar();

#pragma omp target
#pragma omp master
  {
    bar();
  }
#pragma omp target
#pragma omp critical
  {
    bar();
  }
#pragma omp target
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target
#pragma omp task
  {
    bar();
  }
#pragma omp target
  {
#pragma omp taskyield
    bar();
  }
#pragma omp target
  {
#pragma omp barrier
    bar();
  }
#pragma omp target
  {
#pragma omp taskwait
    bar();
  }
#pragma omp target
  {
#pragma omp flush
    bar();
  }
#pragma omp target
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
  {
#pragma omp atomic
    ++a;
  }
#pragma omp target
  {
#pragma omp target // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
  {
#pragma omp teams
    ++a;
  }
#pragma omp target // expected-error {{target construct with nested teams region contains statements outside of the teams construct}}
  {
    ++a;           // expected-note {{statement outside teams construct here}}
#pragma omp teams  // expected-note {{nested teams construct here}}
    ++a;
  }
#pragma omp target
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target
  { 
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target' region}}
  }
#pragma omp target
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target' region}}
  }

// TARGET PARALLEL DIRECTIVE
#pragma omp target parallel
#pragma omp parallel
  bar();
#pragma omp target parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp target parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a target parallel region}}
  {
    bar();
  }
#pragma omp target parallel
#pragma omp single
  bar();

#pragma omp target parallel
#pragma omp master
  {
    bar();
  }
#pragma omp target parallel
#pragma omp critical
  {
    bar();
  }
#pragma omp target parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target parallel
#pragma omp task
  {
    bar();
  }
#pragma omp target parallel
  {
#pragma omp taskyield
    bar();
  }
#pragma omp target parallel
  {
#pragma omp barrier
    bar();
  }
#pragma omp target parallel
  {
#pragma omp taskwait
    bar();
  }
#pragma omp target parallel
  {
#pragma omp flush
    bar();
  }
#pragma omp target parallel
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target parallel
  {
#pragma omp atomic
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp target // expected-error {{region cannot be nested inside 'target parallel' region}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target parallel' region}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target parallel
  {
    ++a;
#pragma omp teams  // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target parallel
  { 
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target parallel
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target parallel' region}}
  }
#pragma omp target parallel
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target parallel' region}}
  }

// TEAMS DIRECTIVE
#pragma omp target
#pragma omp teams
#pragma omp parallel
  bar();
#pragma omp target
#pragma omp teams
#pragma omp for // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp simd // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp sections // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a teams region}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp single // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
  bar();

#pragma omp target
#pragma omp teams
#pragma omp master // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp master' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp critical // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp critical' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp task // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp task' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskyield // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskyield' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp barrier' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskwait // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskwait' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp flush // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp flush' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp atomic // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp atomic' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target enter data' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target exit data' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskloop // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskloop' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp distribute
  for (int j = 0; j < 10; ++j)
    ;        

// TASKLOOP DIRECTIVE
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a taskloop region}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }

#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{region cannot be closely nested inside 'taskloop' region}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'taskloop' region}}
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
// DISTRIBUTE DIRECTIVE
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams  
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp sections
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a distribute region}}
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp single
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp master
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single
      {
	bar();
      }
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
}

void foo() {
  int a = 0;
// PARALLEL DIRECTIVE
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel region}}
  {
    bar();
  }
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp single
  bar();
#pragma omp parallel
#pragma omp master
  bar();
#pragma omp parallel
#pragma omp critical
  bar();
#pragma omp parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp task
  {
    bar();
  }
#pragma omp parallel
  {
#pragma omp taskyield
    bar();
  }
#pragma omp parallel
  {
#pragma omp barrier
    bar();
  }
#pragma omp parallel
  {
#pragma omp taskwait
    bar();
  }
#pragma omp parallel
  {
#pragma omp flush
    bar();
  }
#pragma omp parallel
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel
  {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SIMD DIRECTIVE
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// FOR DIRECTIVE
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a for region}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    bar();
#pragma omp master // expected-error {{region cannot be closely nested inside 'for' region}}
    bar();
#pragma omp critical
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'for' region}}
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // OK
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// FOR SIMD DIRECTIVE
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// SECTIONS DIRECTIVE
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp critical
    bar();
#pragma omp single // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    bar();
#pragma omp master // expected-error {{region cannot be closely nested inside 'sections' region}}
    bar();
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp taskyield
  }
#pragma omp sections
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'sections' region}}
    bar();
  }
#pragma omp sections
  {
#pragma omp taskwait
  }
#pragma omp sections
  {
#pragma omp flush
  }
#pragma omp sections
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp sections
  {
#pragma omp atomic
    ++a;
  }
#pragma omp sections
  {
#pragma omp target
    ++a;
  }
#pragma omp sections
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp sections
  {
#pragma omp target enter data map(to: a)
  }
#pragma omp sections
  {
#pragma omp target exit data map(from: a)
  }
#pragma omp sections
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp sections
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp sections
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SECTION DIRECTIVE
#pragma omp section // expected-error {{orphaned 'omp section' directives are prohibited, it must be closely nested to a sections region}}
  {
    bar();
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp for // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp simd
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a section region}}
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
#pragma omp master // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
#pragma omp critical
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel
      {
#pragma omp single // OK
        {
          bar();
        }
#pragma omp for // OK
        for (int i = 0; i < 10; ++i)
          ;
#pragma omp for simd // OK
        for (int i = 0; i < 10; ++i)
          ;
#pragma omp sections // OK
        {
          bar();
        }
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel for
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel for simd
      for (int i = 0; i < 10; ++i)
        ;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp parallel sections
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp task
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp taskyield
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp taskwait
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp flush
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp atomic
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target parallel
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target enter data map(to: a)
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp target exit data map(from: a)
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
      ++a;
    }
  }
#pragma omp sections
  {
#pragma omp section
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// SINGLE DIRECTIVE
#pragma omp single
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
#pragma omp master // expected-error {{region cannot be closely nested inside 'single' region}}
    bar();
#pragma omp critical
    bar();
  }
#pragma omp single
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp single
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp taskyield
    bar();
  }
#pragma omp single
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'single' region}}
    bar();
  }
#pragma omp single
  {
#pragma omp taskwait
    bar();
  }
#pragma omp single
  {
#pragma omp flush
    bar();
  }
#pragma omp single
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp single
  {
#pragma omp atomic
    ++a;
  }
#pragma omp single
  {
#pragma omp target
    ++a;
  }
#pragma omp single
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp single
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp single
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp single
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp single
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp single
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// MASTER DIRECTIVE
#pragma omp master
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp master // OK, though second 'master' is redundant
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp parallel
    {
#pragma omp master // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp master
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp master
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp master
  {
#pragma omp taskyield
    bar();
  }
#pragma omp master
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'master' region}}
    bar();
  }
#pragma omp master
  {
#pragma omp taskwait
    bar();
  }
#pragma omp master
  {
#pragma omp flush
    bar();
  }
#pragma omp master
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp master
  {
#pragma omp atomic
    ++a;
  }
#pragma omp master
  {
#pragma omp target
    ++a;
  }
#pragma omp master
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp master
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp master
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp master
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp master
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp master
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'master' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// CRITICAL DIRECTIVE
#pragma omp critical
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp master // OK, though second 'master' is redundant
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp parallel
    {
#pragma omp master // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp critical
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp critical
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp critical
  {
#pragma omp taskyield
    bar();
  }
#pragma omp critical
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'critical' region}}
    bar();
  }
#pragma omp critical
  {
#pragma omp taskwait
    bar();
  }
#pragma omp critical(Belka)
  {
#pragma omp critical(Strelka)
    bar();
  }
#pragma omp critical(Tuzik) // expected-note {{previous 'critical' region starts here}}
  {
#pragma omp critical(grelka) // expected-note {{previous 'critical' region starts here}}
    {
#pragma omp critical(Tuzik) // expected-error {{cannot nest 'critical' regions having the same name 'Tuzik'}}
      {
#pragma omp parallel
#pragma omp critical(grelka) // expected-error {{cannot nest 'critical' regions having the same name 'grelka'}}
        {
          bar();
        }
      }
    }
  }
#pragma omp critical
  {
#pragma omp flush
    bar();
  }
#pragma omp critical
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp critical
  {
#pragma omp atomic
    ++a;
  }
#pragma omp critical
  {
#pragma omp target
    ++a;
  }
#pragma omp critical
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp critical
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp critical
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp critical
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp critical
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp critical
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'critical' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// PARALLEL FOR DIRECTIVE
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel for region}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
#pragma omp master // expected-error {{region cannot be closely nested inside 'parallel for' region}}
    {
      bar();
    }
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp master // OK
      {
        bar();
      }
#pragma omp critical // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'parallel for' region}}
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // OK
    bar();
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// PARALLEL FOR SIMD DIRECTIVE
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd// expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
#pragma omp single
      {
        bar();
      }
#pragma omp for
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections
      {
        bar();
      }
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd// expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int j = 0; j < 10; ++j)
      ;
  }

// PARALLEL SECTIONS DIRECTIVE
#pragma omp parallel sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
#pragma omp master // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
#pragma omp critical
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp master // OK
      {
        bar();
      }
#pragma omp critical // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp for simd // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp taskyield
  }
#pragma omp parallel sections
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'parallel sections' region}}
  }
#pragma omp parallel sections
  {
#pragma omp taskwait
  }
#pragma omp parallel sections
  {
#pragma omp flush
  }
#pragma omp parallel sections
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp parallel sections
  {
#pragma omp atomic
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp target enter data map(to: a)
  }
#pragma omp parallel sections
  {
#pragma omp target exit data map(from: a)
  }
#pragma omp parallel sections
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp parallel sections
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// TASK DIRECTIVE
#pragma omp task
#pragma omp for // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp sections // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp task
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a task region}}
  {
    bar();
  }
#pragma omp task
#pragma omp single // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
  bar();
#pragma omp task
#pragma omp master // expected-error {{region cannot be closely nested inside 'task' region}}
  bar();
#pragma omp task
#pragma omp critical
  bar();
#pragma omp task
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp task
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp task
#pragma omp task
  {
    bar();
  }
#pragma omp task
  {
#pragma omp taskyield
    bar();
  }
#pragma omp task
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'task' region}}
    bar();
  }
#pragma omp task
  {
#pragma omp taskwait
    bar();
  }
#pragma omp task
  {
#pragma omp flush
    bar();
  }
#pragma omp task
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp task
  {
#pragma omp atomic
    ++a;
  }
#pragma omp task
  {
#pragma omp target
    ++a;
  }
#pragma omp task
  {
#pragma omp target parallel
    ++a;
  }
#pragma omp task
  {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp task
  {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp task
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp task
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp task
  {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// ATOMIC DIRECTIVE
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp for simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp master // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp critical // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel for simd // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp task // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    {
      bar();
    }
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskyield // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp barrier // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskwait // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp flush // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp atomic // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target parallel // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target enter data map(to: a) // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp target exit data map(from: a) // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp teams // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp taskloop // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an l-value expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp distribute // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// TARGET DIRECTIVE
#pragma omp target
#pragma omp parallel
  bar();
#pragma omp target
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp sections
  {
    bar();
  }
#pragma omp target
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a target region}}
  {
    bar();
  }
#pragma omp target
#pragma omp single
  bar();

#pragma omp target
#pragma omp master
  {
    bar();
  }
#pragma omp target
#pragma omp critical
  {
    bar();
  }
#pragma omp target
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target
#pragma omp task
  {
    bar();
  }
#pragma omp target
  {
#pragma omp taskyield
    bar();
  }
#pragma omp target
  {
#pragma omp barrier
    bar();
  }
#pragma omp target
  {
#pragma omp taskwait
    bar();
  }
#pragma omp target
  {
#pragma omp flush
    bar();
  }
#pragma omp target
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
  {
#pragma omp atomic
    ++a;
  }
#pragma omp target
  {
#pragma omp target // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target' region}}
  }
#pragma omp target
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target' region}}
  }
#pragma omp target
  {
#pragma omp teams
    ++a;
  }
#pragma omp target // expected-error {{target construct with nested teams region contains statements outside of the teams construct}}
  {
    ++a;          // expected-note {{statement outside teams construct here}}
#pragma omp teams // expected-note {{nested teams construct here}}
    ++a;
  }
#pragma omp target
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target
  { 
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'target' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }

// TARGET PARALLEL DIRECTIVE
#pragma omp target parallel
#pragma omp parallel
  bar();
#pragma omp target parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp target parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a target parallel region}}
  {
    bar();
  }
#pragma omp target parallel
#pragma omp single
  bar();

#pragma omp target parallel
#pragma omp master
  {
    bar();
  }
#pragma omp target parallel
#pragma omp critical
  {
    bar();
  }
#pragma omp target parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target parallel
#pragma omp task
  {
    bar();
  }
#pragma omp target parallel
  {
#pragma omp taskyield
    bar();
  }
#pragma omp target parallel
  {
#pragma omp barrier
    bar();
  }
#pragma omp target parallel
  {
#pragma omp taskwait
    bar();
  }
#pragma omp target parallel
  {
#pragma omp flush
    bar();
  }
#pragma omp target parallel
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target parallel
  {
#pragma omp atomic
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp target // expected-error {{region cannot be nested inside 'target parallel' region}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target parallel' region}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target parallel
  {
    ++a;
#pragma omp teams  // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target parallel
  {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target parallel
  { 
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'target parallel' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target parallel
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target parallel' region}}
  }
#pragma omp target parallel
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target parallel' region}}
  }

// TEAMS DIRECTIVE
#pragma omp target
#pragma omp teams
#pragma omp parallel
  bar();
#pragma omp target
#pragma omp teams
#pragma omp for // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp simd // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp sections // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a teams region}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp single // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
  bar();

#pragma omp target
#pragma omp teams
#pragma omp master // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp master' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp critical // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp critical' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp parallel sections
  {
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp task // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp task' directive into a parallel region?}}
  {
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskyield // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskyield' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp barrier' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskwait // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskwait' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp flush // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp flush' directive into a parallel region?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp atomic // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp atomic' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target' directive into a parallel region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target enter data' directive into a parallel region?}}
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp target exit data' directive into a parallel region?}}
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp target
#pragma omp teams
  {
#pragma omp taskloop // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp taskloop' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i)
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp distribute
  for (int j = 0; j < 10; ++j)
    ;

// TASKLOOP DIRECTIVE
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp for simd' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a taskloop region}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }

#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp master // expected-error {{region cannot be closely nested inside 'taskloop' region}}
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'taskloop' region}}
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a)
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a)
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'taskloop' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop
  for (int i = 0; i < 10; ++i)
    ++a;
  }

// DISTRIBUTE DIRECTIVE
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp distribute // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp distribute' directive into a teams region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams  
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp sections
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a distribute region}}
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp single
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp master
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp critical
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single
      {
	bar();
      }
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp task
    {
      bar();
    }
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp taskyield
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp taskwait
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp flush
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp ordered' directive into a for or a parallel for region with 'ordered' clause?}}
    bar();
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target parallel // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp teams // expected-error {{region cannot be closely nested inside 'distribute' region; perhaps you forget to enclose 'omp teams' directive into a target region?}}
    ++a;
  }
  return foo<int>();
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target enter data map(to: a) // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
#pragma omp target
#pragma omp teams
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp target exit data map(from: a) // expected-error {{region cannot be nested inside 'target' region}}
    ++a;
  }
}
