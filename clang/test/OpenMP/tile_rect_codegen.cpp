// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -ast-print %s | FileCheck %s
// expected-no-diagnostics

// -ast-print round-trip tests: a `#pragma omp tile` directive and its associated
// loop must print back as written. This checks parsing/printing only; for the
// generated IR (min-bounded lowering) see tile_rect_codegen_ir.cpp.

extern "C" void body(...) {}

// CHECK-LABEL: void rect_tile_1d(
void rect_tile_1d() {
  // Partial last tile (trip count 4, tile 5); -ast-print prints the original loop.
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
  // Exact-dividing trip count (10, tile 5); -ast-print prints the original loop.
  // CHECK: #pragma omp tile sizes(5)
  // CHECK-NEXT: for (int i = 0; i < 10; i += 1)
  #pragma omp tile sizes(5)
  for (int i = 0; i < 10; i += 1)
    body(i);
}

// CHECK-LABEL: void rect_tile_nested_body_loop(
void rect_tile_nested_body_loop(int n) {
  // The inner `j` loop lives in the tiled construct's body; -ast-print shows the
  // directive and the outer `i` loop printed back.
  // CHECK: #pragma omp tile sizes(4)
  // CHECK-NEXT: for (int i = 0; i < 6; i += 1)
  #pragma omp tile sizes(4)
  for (int i = 0; i < 6; i += 1)
    for (int j = 0; j < n; ++j)
      body(i, j);
}
