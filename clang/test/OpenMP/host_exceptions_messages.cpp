// RUN: %clang_cc1 -verify=expected,host -fopenmp -fsyntax-only %s
// RUN: %clang_cc1 -verify=expected,simd -fopenmp-simd -fsyntax-only %s
// RUN: %clang_cc1 -verify=expected,host -fopenmp -fopenmp-targets=x86_64 -triple x86_64 -fsyntax-only %s

// Exceptions are disabled, so a host compilation must diagnose 'try' and 'throw'
// whether or not an offload target is configured.

void foo();

void bar() {
  try { // expected-error {{cannot use 'try' with exceptions disabled}}
    foo();
  } catch (...) {
  }
}

void baz(bool b) {
  if (b)
    throw 1; // expected-error {{cannot use 'throw' with exceptions disabled}}
}

// A 'device_type(nohost)' function is not emitted by a host -fopenmp compilation, but
// -fopenmp-simd ignores 'declare target' and does emit it.
#pragma omp begin declare target device_type(nohost)
void devonly() {
  try { // simd-error {{cannot use 'try' with exceptions disabled}}
    foo();
  } catch (...) {
  }
}
#pragma omp end declare target

// Same, but marked 'nohost' only after the body has been parsed.
void late() {
  throw 1; // simd-error {{cannot use 'throw' with exceptions disabled}}
}
#pragma omp declare target to(late) device_type(nohost)
