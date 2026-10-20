// RUN: %clang_cc1 -verify=expected -fopenmp -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

void foo() {}

template <class T, typename S, int N>
T tmain(T argc, S **argv) {
  // Correct usages
  #pragma omp parallel severity(fatal)
  #pragma omp parallel severity(warning)

  // Missing parentheses
  #pragma omp parallel severity // expected-error {{expected '(' after 'severity'}}

  // Empty parentheses
  #pragma omp parallel severity() // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  // Invalid value
  #pragma omp parallel severity(error) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}
  #pragma omp parallel severity(unknown) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  // Multiple arguments
  #pragma omp parallel severity(fatal, warning) // expected-error {{expected ')'}} expected-note {{to match this '('}}

  // Unterminated clause
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp parallel severity(fatal

  // Extra tokens after clause
  #pragma omp parallel severity(fatal) extra // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}

  // Multiple severity clauses
  #pragma omp parallel severity(fatal) severity(warning) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'severity' clause}}

  // Severity clause with other clauses (should be valid)
  #pragma omp parallel severity(warning) num_threads(2)

  // Severity clause with invalid clause
  #pragma omp parallel severity(fatal) invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}

  // Severity clause with macro that is not a valid value
  #define NOT_A_SEVERITY 123
  #pragma omp parallel severity(NOT_A_SEVERITY) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  // Severity clause with macro that is a valid value
  #define FATAL fatal
  #pragma omp parallel severity(FATAL)

  // Severity clause with template parameter that is not a valid value
  #pragma omp parallel severity(N) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  foo();

  return argc;
}

int main(int argc, char **argv) {
  // Correct usage
  #pragma omp parallel severity(fatal)

  // Invalid: missing value
  #pragma omp parallel severity() // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  // Invalid: non-keyword
  #pragma omp parallel severity(argc) // expected-error {{expected 'fatal' or 'warning' in OpenMP clause 'severity'}}

  foo();

  return tmain<int, char, 3>(argc, argv);
}
