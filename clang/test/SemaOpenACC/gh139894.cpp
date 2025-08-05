// RUN: %clang_cc1 %s -fopenacc -verify

// Ensure that these don't assert, they previously assumed that their directive
// kind would be valid, but we should make sure that we handle that gracefully
// in cases where they don't.

// expected-error@+1{{invalid OpenACC directive 'foo'}}
#pragma acc foo gang(1)

// expected-error@+1{{invalid OpenACC directive 'foo'}}
#pragma acc foo vector(1)

// expected-error@+1{{invalid OpenACC directive 'foo'}}
#pragma acc foo worker(1)
