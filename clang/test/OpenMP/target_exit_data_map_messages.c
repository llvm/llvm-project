// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp -fopenmp -fno-openmp-extensions -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp -fopenmp -fno-openmp-extensions -ferror-limit 100 -o - -x c++ %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp -fopenmp-simd -fno-openmp-extensions -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp -fopenmp-simd -fno-openmp-extensions -ferror-limit 100 -o - -x c++ %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,ompx -fopenmp -fopenmp-extensions -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,ompx -fopenmp -fopenmp-extensions -ferror-limit 100 -o - -x c++ %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,ompx -fopenmp-simd -fopenmp-extensions -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,ompx -fopenmp-simd -fopenmp-extensions -ferror-limit 100 -o - -x c++ %s -Wuninitialized

int main(int argc, char **argv) {

  int r;
  #pragma omp target exit data // expected-error {{expected at least one 'map' clause for '#pragma omp target exit data'}}

  #pragma omp target exit data map(r) // expected-error {{map type must be specified for '#pragma omp target exit data'}}
  #pragma omp target exit data map(tofrom: r) // expected-error {{map type 'tofrom' is not allowed for '#pragma omp target exit data'}}

  #pragma omp target exit data map(always, from: r) allocate(r) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp target exit data'}}
  #pragma omp target exit data map(delete: r)
  #pragma omp target exit data map(release: r)
  #pragma omp target exit data map(always, alloc: r) // expected-error {{map type 'alloc' is not allowed for '#pragma omp target exit data'}}
  #pragma omp target exit data map(to: r) // expected-error {{map type 'to' is not allowed for '#pragma omp target exit data'}}

  // omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // ompx-error@+1 {{map type modifier 'ompx_hold' is not allowed for '#pragma omp target exit data'}}
  #pragma omp target exit data map(ompx_hold, from: r)
  // omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // ompx-error@+1 {{map type modifier 'ompx_hold' is not allowed for '#pragma omp target exit data'}}
  #pragma omp target exit data map(ompx_hold, release: r)
  // omp-error@+2 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper'}}
  // ompx-error@+1 {{map type modifier 'ompx_hold' is not allowed for '#pragma omp target exit data'}}
  #pragma omp target exit data map(ompx_hold, delete: r)

  return 0;
}
