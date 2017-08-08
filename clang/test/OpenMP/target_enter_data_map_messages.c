// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - -x c++ %s

int main(int argc, char **argv) {

  int r;
  #pragma omp target enter data // expected-error {{expected at least one 'map' clause for '#pragma omp target enter data'}}

  #pragma omp target enter data map(r) // expected-error {{map type must be specified for '#pragma omp target enter data'}}
  #pragma omp target enter data map(tofrom: r) // expected-error {{map type 'tofrom' is not allowed for '#pragma omp target enter data'}}

  #pragma omp target enter data map(always, to: r)
  #pragma omp target enter data map(always, alloc: r)
  #pragma omp target enter data map(always, from: r) // expected-error {{map type 'from' is not allowed for '#pragma omp target enter data'}}
  #pragma omp target enter data map(release: r) // expected-error {{map type 'release' is not allowed for '#pragma omp target enter data'}}
  #pragma omp target enter data map(delete: r) // expected-error {{map type 'delete' is not allowed for '#pragma omp target enter data'}}

  return 0;
}
