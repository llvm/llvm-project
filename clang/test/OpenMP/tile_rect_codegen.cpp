// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -ast-print %s | FileCheck %s
// expected-no-diagnostics

extern "C" void body(...) {}

// CHECK-LABEL: void rect_tile_1d(
void rect_tile_1d() {
  // Tile size 5, trip count 4: 4 % 5 != 0, so predicate is needed.
  // The tile loop should have rectangular bound (the tile size) and
  // the body should be guarded by a validity predicate.
  //
  // CHECK: #pragma omp tile sizes(5)
  // CHECK-NEXT: for (int i = 7; i < 17; i += 3)
  #pragma omp tile sizes(5)
  for (int i = 7; i < 17; i += 3)
    body(i);
}

// CHECK-LABEL: void rect_tile_2d(
void rect_tile_2d() {
  // CHECK: #pragma omp tile sizes(5, 5)
  // CHECK-NEXT: for (int i = 7; i < 17; i += 3)
  // CHECK-NEXT:   for (int j = 7; j < 17; j += 3)
  #pragma omp tile sizes(5, 5)
  for (int i = 7; i < 17; i += 3)
    for (int j = 7; j < 17; j += 3)
      body(i, j);
}

// CHECK-LABEL: void rect_tile_exact_div(
void rect_tile_exact_div() {
  // Tile size 5, trip count 10: 10 % 5 == 0, so predicate is NOT needed.
  // CHECK: #pragma omp tile sizes(5)
  // CHECK-NEXT: for (int i = 0; i < 10; i += 1)
  #pragma omp tile sizes(5)
  for (int i = 0; i < 10; i += 1)
    body(i);
}

// CHECK-LABEL: void rect_tile_nested_body_loop(
void rect_tile_nested_body_loop(int n) {
  // After tiling i, the j loop (from the associated body) should be part
  // of the same canonical nest as the floor and tile loops.
  // CHECK: #pragma omp tile sizes(4)
  // CHECK-NEXT: for (int i = 0; i < 6; i += 1)
  #pragma omp tile sizes(4)
  for (int i = 0; i < 6; i += 1)
    for (int j = 0; j < n; ++j)
      body(i, j);
}
