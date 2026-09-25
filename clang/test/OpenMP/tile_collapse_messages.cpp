// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -fopenmp -fsyntax-only -Wuninitialized -verify %s

// A collapsed intra-tile loop starts from the floor counter of an outer loop
// in the same nest, so that floor must itself be one of the collapsed loops.
// Inside another loop transformation it is assigned in an enclosing loop's
// body and the nest would read a stale value, which is diagnosed rather than
// silently miscomputed.

extern void body(int);

void collapse_through_single_tile() {
#pragma omp for collapse(2)
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    body(i);
}

void collapse_through_tile_2d() {
#pragma omp for collapse(4)
#pragma omp tile sizes(2, 3)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 7; ++j)
      body(i + j);
}

void collapse_outer_of_stacked_tiles() {
  // Only the outer tile's floor and intra-tile loops are collapsed, and that
  // floor is a collapsed counter, so this is fine.
#pragma omp for collapse(2)
#pragma omp tile sizes(2)
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    body(i);
}

void collapse_into_stacked_tiles() {
  // The third loop is the inner tile's intra-tile loop, whose floor is computed
  // inside the outer intra-tile loop's body.
#pragma omp for collapse(3)
#pragma omp tile sizes(2)
#pragma omp tile sizes(4)
  // expected-error@+1 {{cannot collapse the intra-tile loop of a '#pragma omp tile' that is nested inside another loop-transforming directive; OpenMP permits this construct, but it is not yet supported}}
  for (int i = 0; i < 6; ++i)
    body(i);
}

void collapse_simd_over_tile() {
  // Rejected by the pre-existing restriction on nesting inside a 'simd' region.
#pragma omp simd collapse(2)
  // expected-error@+1 {{OpenMP constructs may not be nested inside a simd region except for ordered simd, simd, scan, or atomic directive}}
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    body(i);
}

void collapse_outer_of_stacked_tiles_2d() {
  // Outer tile sizes(2, 3) needs two loops, so it reaches the inner
  // intra-tile loop. Same limitation as tile_of_tile_two_sizes.
#pragma omp for collapse(4)
#pragma omp tile sizes(2, 3)
#pragma omp tile sizes(4)
  // expected-error@+1 {{cannot apply a loop transformation to the intra-tile loop of a '#pragma omp tile' that is nested inside another loop-transforming directive; OpenMP permits this construct, but it is not yet supported}}
  for (int i = 0; i < 6; ++i)
    body(i);
}

void tile_of_tile_two_sizes() {
  // Outer tile sizes(3, 5) needs two loops, so it reaches the inner
  // intra-tile loop. That nest would run the wrong iterations.
#pragma omp tile sizes(3, 5)
#pragma omp tile sizes(2)
  // expected-error@+1 {{cannot apply a loop transformation to the intra-tile loop of a '#pragma omp tile' that is nested inside another loop-transforming directive; OpenMP permits this construct, but it is not yet supported}}
  for (int i = 0; i < 6; ++i)
    body(i);
}
