// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

int main(int argc, char **argv) {

  int r;
  #pragma omp target exit data // expected-error {{expected at least one map clause for '#pragma omp target exit data'}}

  #pragma omp target exit data map(r) // expected-error {{map type must be specified for '#pragma omp target exit data'}}
  #pragma omp target exit data map(tofrom: r) // expected-error {{map type 'tofrom' is not allowed for '#pragma omp target exit data'}}

  #pragma omp target exit data map(always, from: r)
  #pragma omp target exit data map(delete: r)
  #pragma omp target exit data map(release: r)
  #pragma omp target exit data map(always, alloc: r) // expected-error {{map type 'alloc' is not allowed for '#pragma omp target exit data'}}
  #pragma omp target exit data map(to: r) // expected-error {{map type 'to' is not allowed for '#pragma omp target exit data'}}

  return 0;
}
