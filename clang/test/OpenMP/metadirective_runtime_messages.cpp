// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -fsyntax-only %s

// Body diagnostics are reported once per variant.
void body_error_per_variant(int flag) {
#pragma omp metadirective \
  when(user={condition(flag)}: parallel) \
  otherwise(single)
  {
    undeclared = 1; // expected-error 2 {{use of undeclared identifier 'undeclared'}}
  }
}

// Without 'otherwise', the implicit plain-statement variant is parsed too.
void body_error_no_otherwise(int flag) {
#pragma omp metadirective \
  when(user={condition(flag)}: parallel)
  {
    undeclared2 = 1; // expected-error 2 {{use of undeclared identifier 'undeclared2'}}
  }
}

// Variant-specific errors are reported for the variant they apply to.
void variant_specific_errors(int flag) {
#pragma omp metadirective \
  when(user={condition(flag)}: for) \
  otherwise(single)
  for (int i = 0; i < 10; ++i) {
#pragma omp barrier // expected-error {{region cannot be closely nested inside 'for' region}} expected-error {{region cannot be closely nested inside 'single' region}}
  }
}

// A ':' inside the condition (ternary) does not end the selectors.
void ternary_in_condition(int a, int b) {
#pragma omp metadirective \
  when(user={condition(a ? b : 0)}: parallel) \
  otherwise(single)
  {
  }
}

// A 'nothing' variant still consumes the associated statement.
void nothing_variant(int flag) {
#pragma omp metadirective \
  when(user={condition(flag)}: nothing) \
  otherwise(single)
  {
    undeclared3 = 1; // expected-error 2 {{use of undeclared identifier 'undeclared3'}}
  }
}

// Each variant resolves names in the associated statement again.
void local_decl(int flag) {
#pragma omp metadirective \
  when(user={condition(flag)}: parallel) \
  otherwise(single)
  {
    struct L { int v; };
    L l;
    l.v = 1;
  }
}

// A declaration in the shared statement is not visible after it.
struct S { S(); ~S(); void use(); };
void shared_decl_scope(int flag) {
#pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(parallel)
  S s;
  s.use(); // expected-error {{use of undeclared identifier 's'}}
}

// The future spelling warning is reported once.
void future_spelling(int flag) {
#pragma omp parallel
  {
#pragma omp metadirective when(user={condition(flag)}: cancellation_point parallel) otherwise(nothing) // expected-warning {{directive spelling 'cancellation_point' is introduced in a later OpenMP version}}
  }
}
