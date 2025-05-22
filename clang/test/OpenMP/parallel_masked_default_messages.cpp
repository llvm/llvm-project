// RUN: %clang_cc1 -verify -fopenmp-version=50 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-version=50 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -DOMP51 -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -DOMP51 -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
#pragma omp parallel masked default // expected-error {{expected '(' after 'default'}}
  {
#pragma omp parallel masked default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
      {
#pragma omp parallel masked default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
          {
#pragma omp parallel masked default(none // expected-error {{expected ')'}} expected-note {{to match this '('}}
              {
#pragma omp parallel masked default(shared), default(shared) // expected-error {{directive '#pragma omp parallel masked' cannot contain more than one 'default' clause}}
                  {
#pragma omp parallel masked default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
                      {foo();
            }
          }
        }
      }
    }
  }

#pragma omp parallel masked default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  }

#pragma omp parallel masked default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
#pragma omp parallel masked default(shared)
    {
      ++argc;  // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    }
  }

#ifdef OMP51
#pragma omp parallel masked default(firstprivate) // expected-note 2 {{explicit data sharing attribute requested here}}
  {
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#pragma omp parallel masked default(private) // expected-note 2 {{explicit data sharing attribute requested here}}
  {
    ++x; // expected-error {{variable 'x' must have explicitly specified data sharing attributes}}
    ++y; // expected-error {{variable 'y' must have explicitly specified data sharing attributes}}
  }
#endif

  return 0;
}
