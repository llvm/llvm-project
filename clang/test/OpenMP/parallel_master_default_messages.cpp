// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo();

int main(int argc, char **argv) {
#pragma omp parallel master default // expected-error {{expected '(' after 'default'}}
  {
#pragma omp parallel master default( // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    {
#pragma omp parallel master default() // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
      {
#pragma omp parallel master default(none // expected-error {{expected ')'}} expected-note {{to match this '('}}
        {
#pragma omp parallel master default(shared), default(shared) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'default' clause}}
          {
#pragma omp parallel master default(x) // expected-error {{expected 'none' or 'shared' in OpenMP clause 'default'}}
            {
              foo();
            }
          }
        }
      }
    }
  }

#pragma omp parallel master default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  }

#pragma omp parallel master default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
#pragma omp parallel master default(shared)
    {
      ++argc;  // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    }
  }
  return 0;
}
