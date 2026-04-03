// OpenMP split / counts: parse and semantic diagnostics.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s

void body(int);

void parse_and_clause_errors() {

  // Malformed `counts` — missing '('
  // expected-error@+1 {{expected '('}}
  #pragma omp split counts
    ;

  // Empty `counts` list
  // expected-error@+1 {{expected expression}}
  #pragma omp split counts()
    ;

  // Truncated list / missing ')'
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp split counts(3
    for (int i = 0; i < 7; ++i)
      ;

  // Trailing comma only
  // expected-error@+1 {{expected expression}}
  #pragma omp split counts(3,)
    ;

  // Expression after comma missing
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp split counts(3,
    ;

  // Incomplete arithmetic in count (like `tile_messages` sizes(5+))
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp split counts(5+
    ;

  // `for` keyword not a constant-expression operand
  // expected-error@+1 {{expected expression}}
  #pragma omp split counts(for)
    ;

  // Duplicate `counts` clauses
  // expected-error@+1 {{directive '#pragma omp split' cannot contain more than one 'counts' clause}}
  #pragma omp split counts(2, omp_fill) counts(3, omp_fill)
  for (int i = 0; i < 7; ++i)
    ;

  // Disallowed extra clause
  // expected-error@+1 {{unexpected OpenMP clause 'collapse' in directive '#pragma omp split'}}
  #pragma omp split counts(2, omp_fill) collapse(2)
  for (int i = 0; i < 7; ++i)
    ;

  // Non-relational loop condition (canonical loop check)
  #pragma omp split counts(omp_fill)
  // expected-error@+1 {{condition of OpenMP for loop must be a relational comparison ('<', '<=', '>', '>=', or '!=') of loop variable 'i'}}
  for (int i = 0; i / 3 < 7; ++i)
    ;

  // More than one `omp_fill`
  // expected-error@+1 {{exactly one 'omp_fill' must appear in the 'counts' clause}}
  #pragma omp split counts(omp_fill, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);

  // No `omp_fill` at all — also triggers "exactly one" diagnostic.
  // expected-error@+1 {{exactly one 'omp_fill' must appear in the 'counts' clause}}
  #pragma omp split counts(2, 3)
  for (int i = 0; i < 10; ++i)
    body(i);

  // Positive: `omp_fill` may appear at any position in `counts` (not required to be last).
  #pragma omp split counts(omp_fill, 2)
  for (int i = 0; i < 10; ++i)
    body(i);

  // OpenMP 6.0: non-`omp_fill` list items must be integral constant expressions.
  {
    int v = 3; // expected-note {{declared here}}
    #pragma omp split counts(v, omp_fill) // expected-error {{expression is not an integral constant expression}} \
                                            // expected-note {{read of non-const variable 'v' is not allowed in a constant expression}}
    for (int i = 0; i < 10; ++i)
      body(i);
  }
}

void associated_statement_diagnostics() {
  {
    // expected-error@+2 {{expected statement}}
    #pragma omp split counts(omp_fill)
  }

  // Not a `for` loop (contrast `split_diag_errors.c` / `while`)
  // expected-error@+2 {{statement after '#pragma omp split' must be a for loop}}
  #pragma omp split counts(omp_fill)
  int b = 0;

  // expected-warning@+2 {{extra tokens at the end of '#pragma omp split' are ignored}}
  // expected-error@+1 {{directive '#pragma omp split' requires the 'counts' clause}}
  #pragma omp split foo
  for (int i = 0; i < 7; ++i)
    ;
}
