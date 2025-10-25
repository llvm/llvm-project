// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -fopenmp-version=60 -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -fopenmp-version=60 -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

#pragma omp assumes // expected-error {{expected at least one 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism' clause for '#pragma omp assumes'}}
#pragma omp begin // expected-error {{expected an OpenMP directive}}
#pragma omp begin assumes // expected-error {{expected at least one 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism' clause for '#pragma omp begin assumes'}}
#pragma omp end assumes

#pragma omp assumes foobar // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp begin assumes foobar // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp end assumes

#pragma omp begin assumes foobar(foo 2 no_openmp // expected-error {{expected ')'}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{to match this '('}}
#pragma omp assumes foobar(foo 2 no_openmp // expected-error {{expected ')'}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{to match this '('}}
#pragma omp end assumes

#pragma omp begin assumes foobar(foo 2 baz) // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
#pragma omp assumes foobar(foo 2 baz) // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
#pragma omp end assumes

#pragma omp begin assumes foobar foo 2 baz) bar // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp assumes foobar foo 2 baz) bar // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp end assumes

#pragma omp assumes no_openmp(1) // expected-warning {{'no_openmp' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
#pragma omp begin assumes no_openmp(1 2 3) // expected-warning {{'no_openmp' clause should not be followed by arguments; tokens will be ignored}} expected-note {{the ignored tokens spans until here}}
#pragma omp end assumes no_openmp(1)

#pragma omp assumes foobar no_openmp bazbaz // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp begin assumes foobar no_openmp bazbaz // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp end assumes

#pragma omp begin assumes foobar(foo 2 baz) no_openmp bazbaz(foo 2 baz) // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{the ignored tokens spans until here}}
#pragma omp assumes foobar(foo 2 baz) no_openmp bazbaz(foo 2 baz) // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{the ignored tokens spans until here}}
#pragma omp end assumes

#pragma omp begin assumes foobar(foo (2) baz) no_openmp bazbaz(foo (2)) baz) // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{the ignored tokens spans until here}}
#pragma omp assumes foobar(foo () baz) no_openmp bazbaz(foo ((2) baz) // expected-error {{expected ')'}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; tokens will be ignored}} expected-note {{the ignored tokens spans until here}} expected-note {{to match this '('}}
#pragma omp end assumes

#pragma omp assumes no_openmp foobar no_openmp // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp begin assumes no_openmp foobar no_openmp // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp end assumes

#pragma omp assumes holds(1, 2 3)
#pragma omp begin assumes holds(1, 2 3)
#pragma omp end assumes

#pragma omp assumes absent(1, 2 3)
#pragma omp begin assumes absent(1, 2 3)
#pragma omp end assumes

#pragma omp assumes contains(1, 2 3)
#pragma omp begin assumes contains(1, 2 3)
#pragma omp end assumes

#pragma omp assumes ext // expected-warning {{valid assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp begin assumes ext // expected-warning {{valid begin assumes clauses start with 'ext_', 'absent', 'contains', 'holds', 'no_openmp', 'no_openmp_routines', 'no_openmp_constructs', 'no_parallelism'; token will be ignored}}
#pragma omp end assumes

// FIXME: We should be getting an expected note about where the span of ignored
// tokens ends. However, error recovery ends up lexing the 'not' token,
// emitting a (silenced) diagnostic about use of a C++ keyword in C, and the
// note gets associated with *that* (silenced) diagnostic. This is an existing
// issue that also happens with error recovery of reserved identifiers or
// extension tokens, but is unfortunate nonetheless.
#pragma omp assumes ext_123(not allowed) // expected-warning {{'ext_123' clause should not be followed by arguments; tokens will be ignored}}
#pragma omp begin assumes ext_123(not allowed) // expected-warning {{'ext_123' clause should not be followed by arguments; tokens will be ignored}}
#pragma omp end assumes

#pragma omp end assumes // expected-error {{'#pragma omp end assumes' with no matching '#pragma omp begin assumes'}}

// TODO: we should emit a warning at least.
#pragma omp begin assumes ext_abc
