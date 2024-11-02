// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s \
// RUN:   -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s \
// RUN:  -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s \
// RUN:  -fopenmp-version=51 -DOMP51 -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s \
// RUN:  -fopenmp-version=51 -DOMP51 -Wuninitialized

void foo();

int main(int argc, char **argv) {
#pragma omp parallel sections default // expected-error {{expected '(' after 'default'}}
  {
#pragma omp parallel sections default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
      {
#pragma omp parallel sections default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
          {
#pragma omp parallel sections default(none // expected-error {{expected ')'}} expected-note {{to match this '('}}
              {
#pragma omp parallel sections default(shared), default(shared) // expected-error {{directive '#pragma omp parallel sections' cannot contain more than one 'default' clause}}
                  {
#pragma omp parallel sections default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
                      {foo();
            }
          }
        }
      }
    }
  }

#pragma omp parallel sections default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  }

#pragma omp parallel sections default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
#pragma omp parallel sections default(shared)
    {
      ++argc;  // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    }
  }

#ifdef OMP51
#pragma omp parallel sections default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
#pragma omp parallel sections default(private)
    {
      ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    }
  }
#endif // OMP51
  return 0;
}
