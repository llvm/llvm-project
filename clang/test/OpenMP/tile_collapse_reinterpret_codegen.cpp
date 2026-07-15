// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

// Codegen tests for how a loop-associated directive (e.g. 'omp for' with a
// 'collapse' clause) consumes a '#pragma omp tile' intra-tile loop.
//
// These tests pin the key IR invariants of the below design:
// The intra-tile loop is stored in its min-bounded form
//   for (.tile.iv = .floor.iv; .tile.iv < min(.floor.iv + tilesize, N); ++.tile.iv)
// and carries an internal OMPInvariantPredicateBoundAttr hint. When a directive
// consumes the loop, it is reinterpreted as a rectangular loop with a constant
// per-tile trip count (the tile size), and the overshoot of a partial last tile
// is handled by a predicate emitted as a body guard.

extern "C" void body(...) {}

// PRINT-LABEL: void collapse2_tile1d(
// PRINT:       #pragma omp parallel for collapse(2)
// PRINT-NEXT:  #pragma omp tile sizes(2)
// IR-LABEL: define internal void @collapse2_tile1d.omp_outlined
// IR:       %.floor_0.iv{{[0-9a-z.]*}} = alloca i32
// IR:       %.tile_0.iv{{[0-9a-z.]*}} = alloca i32
// IR-NOT:   .tile.cnt
// IR:       omp.inner.for.body:
// IR:       store i32 %{{.*}}, ptr %.tile_0.iv
// IR:       %[[PRED:.*]] = icmp slt i32 %{{.*}}, %{{.*}}
// IR-NEXT:  br i1 %[[PRED]], label %omp.body.next, label %omp.body.continue
// IR:       omp.body.next:
// IR:       call void (...) @body
// IR-NEXT:  br label %omp.body.continue
// IR:       omp.body.continue:
extern "C" void collapse2_tile1d(int n) {
#pragma omp parallel for collapse(2)
#pragma omp tile sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// PRINT-LABEL: void collapse3_tile1d_nested(
// PRINT:       #pragma omp parallel for collapse(3)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define internal void @collapse3_tile1d_nested.omp_outlined
// IR-NOT:   .tile.cnt
// IR:       omp.inner.for.body:
// IR:       store i32 %{{.*}}, ptr %.tile_0.iv
// IR:       %[[PRED:.*]] = icmp slt i32 %{{.*}}, %{{.*}}
// IR-NEXT:  br i1 %[[PRED]], label %omp.body.next, label %omp.body.continue
// IR:       omp.body.next:
// IR:       call void (...) @body
// IR-NEXT:  br label %omp.body.continue
extern "C" void collapse3_tile1d_nested(int n) {
#pragma omp parallel for collapse(3)
#pragma omp tile sizes(5)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}

// PRINT-LABEL: void tile2d_inside_tile1d(
// PRINT:       #pragma omp tile sizes(3)
// PRINT-NEXT:  #pragma omp tile sizes(2)
// IR-LABEL: define {{.*}} @tile2d_inside_tile1d(
// IR:       %.tile_0.iv{{[0-9a-z.]*}} = alloca i32
// IR-NOT:   .tile.cnt
// IR:       cond.true:
// IR:       cond.false:
// IR:       cond.end:
// IR:       %{{.*}} = phi i32
// IR-NOT:   omp.body.next
extern "C" void tile2d_inside_tile1d(int n) {
#pragma omp tile sizes(3)
#pragma omp tile sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// PRINT-LABEL: void collapse3_tile2d(
// PRINT:       #pragma omp parallel for collapse(3)
// PRINT-NEXT:  #pragma omp tile sizes(2, 3)
// IR-LABEL: define internal void @collapse3_tile2d.omp_outlined
// IR-NOT:   .tile.cnt
// IR:       omp.inner.for.body:
// IR:       store i32 %{{.*}}, ptr %.tile_0.iv
// IR:       br i1 %{{.*}}, label %omp.body.next, label %omp.body.continue
// IR:       omp.body.next:
// IR:       call void (...) @body
extern "C" void collapse3_tile2d(int n) {
#pragma omp parallel for collapse(3)
#pragma omp tile sizes(2, 3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}

// PRINT-LABEL: void tile1d_constant_div_no_predicate(
// PRINT:       #pragma omp tile sizes(5)
// IR-LABEL: define {{.*}} @tile1d_constant_div_no_predicate(
// IR:       %.floor_0.iv{{[0-9a-z.]*}} = alloca i32
// IR:       %.tile_0.iv{{[0-9a-z.]*}} = alloca i32
// IR-NOT:   .tile.cnt
// IR-NOT:   omp.body.next
// IR:       call void (...) @body
extern "C" void tile1d_constant_div_no_predicate() {
#pragma omp tile sizes(5)
  for (int i = 0; i < 10; ++i)
    body(i);
}
