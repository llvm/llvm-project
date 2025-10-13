// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=45 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=51 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=52 -verify=expected,omp-52-and-earlier -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp -fopenmp-version=60 -verify=expected,omp-60-and-later   -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=45 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=51 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=52 -verify=expected,omp-52-and-earlier  -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fopenmp-simd -fopenmp-version=60 -verify=expected,omp-60-and-later    -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

int main(int argc, char **argv) {
  int i, z;
  #pragma omp target simd nowait( // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}} omp-60-and-later-error {{expected expression}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait device (-10u)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (3.14) device (-10u) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (argc>> z) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (argv[1] = 2) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}} omp-60-and-later-error {{expected ')'}} omp-60-and-later-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (argc > 0 ? argv[1] : argv[2]) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd nowait (S1) // omp-52-and-earlier-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}} omp-60-and-later-error {{use of undeclared identifier 'S1'}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
