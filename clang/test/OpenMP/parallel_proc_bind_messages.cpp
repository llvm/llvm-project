// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,lt51 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,ge51 -fopenmp -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,lt50,lt51 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,lt51 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,ge50,ge51 -fopenmp-simd -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

void foo();

int main(int argc, char **argv) {
  #pragma omp parallel proc_bind // expected-error {{expected '(' after 'proc_bind'}}
  #pragma omp parallel proc_bind ( // ge51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}} lt51-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel proc_bind () // ge51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}} lt51-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  #pragma omp parallel proc_bind (master // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel proc_bind (close), proc_bind(spread) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'proc_bind' clause}}
  #pragma omp parallel proc_bind (x) // ge51-error {{expected 'master', 'close', 'spread' or 'primary' in OpenMP clause 'proc_bind'}} lt51-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}} 
  foo();

  #pragma omp parallel proc_bind(master)
  ++argc;

  #pragma omp parallel proc_bind(close)
  #pragma omp parallel proc_bind(spread)
  ++argc;

  #pragma omp parallel proc_bind(primary) // lt51-error {{expected 'master', 'close' or 'spread' in OpenMP clause 'proc_bind'}}
  ++argc;
  return 0;
}
