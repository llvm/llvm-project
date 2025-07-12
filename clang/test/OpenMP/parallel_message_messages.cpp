// RUN: %clang_cc1 -verify=expected -fopenmp -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

void foo() {}

template <class T, typename S, int N>
T tmain(T argc, S **argv) {
  // Correct usage
  #pragma omp parallel message("correct message")

  // Missing parentheses
  #pragma omp parallel message // expected-error {{expected '(' after 'message'}}
  
  // Empty parentheses
  #pragma omp parallel message() // expected-error {{expected expression}}

  // Non-string literal
  #pragma omp parallel message(123) // expected-warning {{expected string literal in 'clause message' - ignoring}}
  #pragma omp parallel message(argc) // expected-warning {{expected string literal in 'clause message' - ignoring}}
  #pragma omp parallel message(argv[0]) // expected-warning {{expected string literal in 'clause message' - ignoring}}

  // Multiple arguments
  #pragma omp parallel message("msg1", "msg2") // expected-error {{expected ')'}} expected-note {{to match this '('}}
  
  // Unterminated string
  // expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-warning@+1 {{missing terminating '"' character}} expected-note@+1 {{to match this '('}}
  #pragma omp parallel message("unterminated

  // Unterminated clause
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp parallel message("msg"

  // Extra tokens after clause
  #pragma omp parallel message("msg") extra // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}

  // Multiple message clauses
  #pragma omp parallel message("msg1") message("msg2") // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'message' clause}}

  // Message clause with other clauses (should be valid, but test for interaction)
  #pragma omp parallel message("msg") num_threads(2)

  // Message clause with invalid clause
  #pragma omp parallel message("msg") invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}

  // Message clause with missing string and other clause
  #pragma omp parallel message() num_threads(2) // expected-error {{expected expression}}

  // Message clause with macro that is not a string
  #define NOT_A_STRING 123
  #pragma omp parallel message(NOT_A_STRING) // expected-warning {{expected string literal in 'clause message' - ignoring}}

  // Message clause with template parameter that is not a string
  #pragma omp parallel message(N) // expected-warning {{expected string literal in 'clause message' - ignoring}}

  // Message clause with macro that is a string
  #define A_STRING "macro string"
  #pragma omp parallel message(A_STRING)

  // Message clause with concatenated string literals
  #pragma omp parallel message("hello" " world")

  // Message clause with wide string literal
  #pragma omp parallel message(L"wide string")

  // Message clause with UTF-8 string literal
  #pragma omp parallel message(u8"utf8 string")

  // Message clause with raw string literal
  #pragma omp parallel message(R"(raw string)")

  foo();

  return argc;
}

int main(int argc, char **argv) {
  // Correct usage
  #pragma omp parallel message("main correct")

  // Invalid: missing string
  #pragma omp parallel message() // expected-error {{expression}}

  // Invalid: non-string
  #pragma omp parallel message(argc) // expected-warning {{expected string literal in 'clause message' - ignoring}}

  foo();

  return tmain<int, char, 3>(argc, argv);
}
