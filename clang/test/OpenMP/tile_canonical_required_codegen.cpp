// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// expected-no-diagnostics

// Positive tests for canonical intra-tile loop shape.

extern "C" void body(...) {}

// PRINT-LABEL: void collapse2_tile1d(
// IR-LABEL: define {{.*}} @collapse2_tile1d(
// collapse(2) consumes floor+tile; predicate stays as a body guard.
// IR should have one flattened loop body and no nested floor/tile loop.
// IR: %.tile.cnt.0.iv.i
// IR: omp.inner.for.body:
// IR: br i1 %{{.*}}, label %if.then, label %if.end
// IR: if.then:
// IR: call void (...) @body
// IR-NEXT: br label %if.end
// IR: if.end:
extern "C" void collapse2_tile1d(int n) {
#pragma omp parallel for collapse(2)
#pragma omp tile sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// PRINT-LABEL: void collapse3_tile1d_nested(
// IR-LABEL: define {{.*}} @collapse3_tile1d_nested(
// collapse(3) reaches through the tile guard to include `j`.
// IR: br i1 %{{.*}}, label %omp_tile.pred.then, label %omp_tile.pred.end
// IR: omp_tile.pred.then:
// IR: call void (...) @body
// IR-NEXT: br label %omp_tile.pred.end
// IR: omp_tile.pred.end:
extern "C" void collapse3_tile1d_nested(int n) {
#pragma omp parallel for collapse(3)
#pragma omp tile sizes(5)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}

// PRINT-LABEL: void tile2d_inside_tile1d(
// IR-LABEL: define {{.*}} @tile2d_inside_tile1d(
// Nested tiling: both intra-tile loops stay canonical and hoist their predicates.
// IR: %.tile.cnt.0.iv..floor_0.iv.i = alloca i32
// IR: %.tile.cnt.0.iv.i = alloca i32
// IR: %omp_tile.invariant_pred_bound = and i1
// IR: %omp_tile.invariant_pred_bound{{[0-9]+}} = and i1
extern "C" void tile2d_inside_tile1d(int n) {
#pragma omp tile sizes(3)
#pragma omp tile sizes(2)
  for (int i = 0; i < n; ++i)
    body(i);
}

// PRINT-LABEL: void collapse3_tile2d(
// IR-LABEL: define {{.*}} @collapse3_tile2d(
// 2-D tile + collapse(3): dim checks combine and are hoisted into the surviving loop.
// IR: %.tile.cnt.0.iv.i
// IR: %.tile.cnt.1.iv.j
// IR: land.rhs:
// IR: land.end:
// IR: %omp_tile.invariant_pred_bound = and i1
extern "C" void collapse3_tile2d(int n) {
#pragma omp parallel for collapse(3)
#pragma omp tile sizes(2, 3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}

// PRINT-LABEL: void tile1d_constant_div_no_predicate(
// IR-LABEL: define {{.*}} @tile1d_constant_div_no_predicate(
// Exact-divisible case: canonical tile-count loop, no predicate.
// IR: %.tile.cnt.0.iv.i
// IR: for.cond1:
// IR: %[[TC:.*]] = icmp slt i32 %{{.*}}, 5
// IR-NEXT: br i1 %[[TC]], label %for.body3, label %for.end
// IR-NOT: omp_tile.invariant_pred_bound
// IR-NOT: omp_tile.pred
extern "C" void tile1d_constant_div_no_predicate() {
#pragma omp tile sizes(5)
  for (int i = 0; i < 10; ++i)
    body(i);
}
