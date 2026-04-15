// Regression tests for rectangular #pragma omp tile lowering:
// - Inner loop counter .tile.cnt.* with bound == tile size (constant).
// - Validity guard (icmp + branch) when remainder iterations exist or trip count
//   is not proven divisible at compile time.
//
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

extern "C" void body(int);

// Trip count 6, tile 4 → remainder; body must be guarded.
// IR-LABEL: remainder_6_tile_4v(
void remainder_6_tile_4(void) {
  // Rectangular tile counter and constant tile-size bound.
  // IR-DAG: %.tile.cnt{{.*}} = alloca i32
  // IR: icmp {{.*}} i32 {{.*}}, 4
  // IR: icmp {{.*}} i32 {{.*}}, 6
  // IR: br i1 {{.*}}, label %if.then{{.*}}, label %if.end{{.*}}
  // IR: call {{.*}} @body(
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    body(i);
}

// Simple stride-1 loop; tile size matches full tiles only (10, tile 5).
// IR-LABEL: full_tiles_10_5v(
void full_tiles_10_5(void) {
  // IR-DAG: %.tile.cnt{{.*}} = alloca i32
  // IR: icmp {{.*}} i32 {{.*}}, 5
  // IR: call {{.*}} @body(
#pragma omp tile sizes(5)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// Variable tile size: clamp (TS<=0 ? 1 : TS) is inlined; bound compares to %cond.
// IR-LABEL: var_tilei(
void var_tile(int ts) {
  // IR-DAG: %.tile.cnt{{.*}} = alloca i32
  // IR: %cond = phi i32
  // IR: icmp {{.*}} i32 {{.*}}, %cond
  // IR: call {{.*}} @body(
#pragma omp tile sizes(ts)
  for (int i = 0; i < 12; ++i)
    body(i);
}

// Two tiled dimensions → two tile counters.
// IR-LABEL: two_dimv(
void two_dim(void) {
  // IR-DAG: %.tile.cnt.0.iv.i{{.*}} = alloca i32
  // IR-DAG: %.tile.cnt.1.iv.j{{.*}} = alloca i32
  // IR: icmp {{.*}} i32 {{.*}}, 2
  // IR: icmp {{.*}} i32 {{.*}}, 2
  // IR: call {{.*}} @body(
#pragma omp tile sizes(2, 2)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      body(i + j);
}

// Nested loop in tile body: outer nest still uses .tile.cnt for tiled dim.
// IR-LABEL: nested_inner_ji(
void nested_inner_j(int n) {
  // IR-DAG: %.tile.cnt{{.*}} = alloca i32
  // IR: icmp {{.*}} i32 {{.*}}, 4
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < n; ++j)
      body(i + j);
}

// Predicate elision: when trip count is evenly divisible by constant tile size,
// no if-guard is needed (no icmp against trip count in tile body).
// IR-LABEL: elided_predicatev(
void elided_predicate(void) {
  // Trip count 12, tile size 4 → 12 % 4 == 0, no predicate.
  // IR-NOT: if.then
  // IR-NOT: if.end
  // IR: call {{.*}} @body(
#pragma omp tile sizes(4)
  for (int i = 0; i < 12; ++i)
    body(i);
}
