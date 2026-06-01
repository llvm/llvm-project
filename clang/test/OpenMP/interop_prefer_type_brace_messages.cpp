// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -std=c++11 -o - %s

// Diagnostics for the OMP 6.0 brace-grouped prefer_type syntax.

typedef void *omp_interop_t;

static void foo() {
  omp_interop_t obj;

  // expected-error@+2 {{only one 'fr' selector allowed per preference-specification}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({fr("sycl"), fr("opencl")}), targetsync: obj)

  // Non-constant variable expression inside fr().
  int x = 2;
  // expected-error@+2 {{prefer_list item must be a string literal or constant integral expression}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({fr(x)}), targetsync: obj)

  // Floating-point literal inside fr().
  // expected-error@+2 {{prefer_list item must be a string literal or constant integral expression}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({fr(1.5)}), targetsync: obj)

  // Integer inside attr() — only string literals allowed.
  // expected-error@+2 {{expected <string_literal>}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({attr(1)}), targetsync: obj)

  // Empty attr() — at least one ext-string-literal is required.
  // expected-error@+2 {{expected <string_literal>}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({attr()}), targetsync: obj)

  // expected-error@+2 {{expected '(' after 'fr'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({fr "sycl"}), targetsync: obj)

  // expected-error@+2 {{expected '(' after 'attr'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({attr "ompx_propX"}), targetsync: obj)

  // Anything that is not 'fr' or 'attr' is rejected by the brace parser.
  // expected-error@+2 {{expected 'fr' or 'attr' selector in 'prefer_type'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({foo}), targetsync: obj)

  // A non-identifier where a selector is expected is rejected too.
  // expected-error@+2 {{expected 'fr' or 'attr' selector in 'prefer_type'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({42}), targetsync: obj)

  // An empty pref-spec '{}' requires at least one 'fr'/'attr' selector.
  // expected-error@+2 {{expected 'fr' or 'attr' selector in 'prefer_type'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type({}), targetsync: obj)

  // An empty prefer_type() requires at least one preference-specification.
  // expected-error@+2 {{expected preference-specification in 'prefer_type'}}
  // expected-error@+1 {{expected at least one 'init', 'use', 'destroy', or 'nowait' clause for '#pragma omp interop'}}
  #pragma omp interop init(prefer_type(), targetsync: obj)
}
