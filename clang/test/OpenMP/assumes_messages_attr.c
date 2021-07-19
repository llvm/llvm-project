// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -fopenmp-version=51 -std=c99 -fms-extensions -fdouble-square-bracket-attributes -Wno-pragma-pack %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -fopenmp-version=51 -std=c99 -fms-extensions -fdouble-square-bracket-attributes -Wno-pragma-pack %s

[[omp::directive(assumes)]]; // expected-error {{expected at least one 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism' clause for '#pragma omp assumes'}}
[[omp::directive(begin)]]; // expected-error {{expected an OpenMP directive}}
[[omp::directive(begin assumes)]]; // expected-error {{expected at least one 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism' clause for '#pragma omp begin assumes'}}
[[omp::directive(end assumes)]];

[[omp::directive(assumes foobar)]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(begin assumes foobar)]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(end assumes)]];

[[omp::directive(begin assumes foobar(foo 2 baz))]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(assumes foobar(foo 2 baz))]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(end assumes)]];

[[omp::directive(assumes no_openmp(1))]]; // expected-warning {{'no_openmp' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(begin assumes no_openmp(1 2 3))]]; // expected-warning {{'no_openmp' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(end assumes no_openmp(1))]];

[[omp::directive(assumes foobar no_openmp bazbaz)]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(begin assumes foobar no_openmp bazbaz)]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(end assumes)]];

[[omp::directive(begin assumes foobar(foo 2 baz) no_openmp bazbaz(foo 2 baz))]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(assumes foobar(foo 2 baz) no_openmp bazbaz(foo 2 baz))]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(end assumes)]];

[[omp::directive(assumes no_openmp foobar no_openmp)]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(begin assumes no_openmp foobar no_openmp)]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(end assumes)]];

[[omp::directive(assumes holds(1, 2 3))]];
[[omp::directive(begin assumes holds(1, 2 3))]];
[[omp::directive(end assumes)]];

[[omp::directive(assumes absent(1, 2 3))]];
[[omp::directive(begin assumes absent(1, 2 3))]];
[[omp::directive(end assumes)]];

[[omp::directive(assumes contains(1, 2 3))]];
[[omp::directive(begin assumes contains(1, 2 3))]];
[[omp::directive(end assumes)]];

[[omp::directive(assumes ext)]]; // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(begin assumes ext)]]; // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_parallelism'; token will be ignored}}
[[omp::directive(end assumes)]];

[[omp::directive(assumes ext_123(not allowed))]]; // expected-warning {{'ext_123' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(begin assumes ext_123(not allowed))]]; // expected-warning {{'ext_123' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
[[omp::directive(end assumes)]];

[[omp::directive(end assumes)]]; // expected-error {{'#pragma omp end assumes' with no matching '#pragma omp begin assumes'}}

// TODO: we should emit a warning at least.
[[omp::directive(begin assumes ext_abc)]];

