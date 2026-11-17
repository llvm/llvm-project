// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=IR

// Check same results after serialization round-trip, which exercises reading
// and writing OMPInvariantPredicateBoundAttr including its optional predicate.
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fclang-abi-compat=latest -std=c++20 -fopenmp -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

// expected-no-diagnostics

// How a loop-associated directive consumes a '#pragma omp tile': the stored
// min-bounded intra-tile loop is reinterpreted as a rectangular one with a
// constant trip count (the tile size), and a partial last tile is handled by a
// body guard. Worksharing, 'distribute' and 'ordered(n)' consumers each read
// the iteration space differently, so all three are covered. The hint is
// internal and never dumped, so it is observed through the guard it produces
// ('omp.body.next') and through the PCH round-trip above.

#ifndef HEADER
#define HEADER

extern "C" void body(...) {}

// PRINT-LABEL: void collapse2_tile1d(
// PRINT:       #pragma omp parallel for collapse(2)
// PRINT-NEXT:  #pragma omp tile sizes(2)
// IR-LABEL: define internal void @collapse2_tile1d.omp_outlined
// IR:       %.floor_0.iv{{[0-9a-z.]*}} = alloca i32
// IR:       %.tile_0.iv{{[0-9a-z.]*}} = alloca i32
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

// A non-tiled inner loop joins the collapsed nest as a third counter, and the
// guard still applies to the tile counter alone.
// PRINT-LABEL: void collapse3_tile1d_nested(
// PRINT:       #pragma omp parallel for collapse(3)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define internal void @collapse3_tile1d_nested.omp_outlined
// IR:       omp.inner.for.body:
// IR:       store i32 %{{.*}}, ptr %.tile_0.iv
// IR:       store i32 %{{.*}}, ptr %j
// IR:       %[[TILEIV:.*]] = load i32, ptr %.tile_0.iv
// IR:       %[[PRED:.*]] = icmp slt i32 %[[TILEIV]], %{{.*}}
// IR-NEXT:  br i1 %[[PRED]], label %omp.body.next, label %omp.body.continue
extern "C" void collapse3_tile1d_nested(int n) {
#pragma omp parallel for collapse(3)
#pragma omp tile sizes(5)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      body(i, j);
}

// An enclosing 'tile' is another loop transformation rather than a consumer, so
// the inner intra-tile loop stays min-bounded and gets no guard.
// PRINT-LABEL: void tile2d_inside_tile1d(
// PRINT:       #pragma omp tile sizes(3)
// PRINT-NEXT:  #pragma omp tile sizes(2)
// IR-LABEL: define {{.*}} @tile2d_inside_tile1d(
// IR:       %.tile_0.iv{{[0-9a-z.]*}} = alloca i32
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

// Trip count 10 is a multiple of the tile size 5, so no tile is partial, the
// hint carries no predicate, and the body is emitted with no guard.
// PRINT-LABEL: void collapse2_tile1d_no_partial_tile(
// PRINT:       #pragma omp parallel for collapse(2)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define internal void @collapse2_tile1d_no_partial_tile.omp_outlined
// IR:       omp.inner.for.body:
// IR-NOT:   omp.body.next
// IR:       call void (...) @body
// IR:       omp.body.continue:
extern "C" void collapse2_tile1d_no_partial_tile() {
#pragma omp parallel for collapse(2)
#pragma omp tile sizes(5)
  for (int i = 0; i < 10; ++i)
    body(i);
}

// Trip count 11 is not a multiple of the tile size 5, so the predicate is kept
// even though both values are compile-time constants.
// PRINT-LABEL: void collapse2_tile1d_constant_partial_tile(
// PRINT:       #pragma omp parallel for collapse(2)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define internal void @collapse2_tile1d_constant_partial_tile.omp_outlined
// IR:       omp.inner.for.body:
// IR:       br i1 %{{.*}}, label %omp.body.next, label %omp.body.continue
// IR:       omp.body.next:
// IR:       call void (...) @body
extern "C" void collapse2_tile1d_constant_partial_tile() {
#pragma omp parallel for collapse(2)
#pragma omp tile sizes(5)
  for (int i = 0; i < 11; ++i)
    body(i);
}

// 'distribute' computes bounds through its own LB/UB/EUB/DistCond fields, so
// check that the reinterpretation and its guard survive that path too.
// PRINT-LABEL: void dist_collapse2_tile1d(
// PRINT:       #pragma omp teams distribute collapse(2)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define internal void @dist_collapse2_tile1d.omp_outlined
// IR:       omp.inner.for.body:
// IR:       store i32 %{{.*}}, ptr %.tile_0.iv
// IR:       br i1 %{{.*}}, label %omp.body.next, label %omp.body.continue
// IR:       omp.body.next:
// IR:       call void (...) @body
extern "C" void dist_collapse2_tile1d(int n) {
#pragma omp teams distribute collapse(2)
#pragma omp tile sizes(5)
  for (int i = 0; i < n; ++i)
    body(i);
}

// 'ordered(n)' feeds each collapsed loop's trip count to __kmpc_doacross_init,
// so the intra-tile dimension must report the constant tile size, matching the
// space 'collapse' linearizes.
// PRINT-LABEL: void ordered_collapse2_tile1d(
// PRINT:       #pragma omp for collapse(2) ordered(2)
// PRINT-NEXT:  #pragma omp tile sizes(5)
// IR-LABEL: define {{.*}} @ordered_collapse2_tile1d(
// IR:       %[[DIM1:.*]] = getelementptr inbounds [2 x %struct.kmp_dim], ptr %dims, i64 0, i64 1
// IR:       %[[UP1:.*]] = getelementptr inbounds nuw %struct.kmp_dim, ptr %[[DIM1]], i32 0, i32 1
// IR-NEXT:  store i64 5, ptr %[[UP1]]
// IR:       call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.*}}, i32 2, ptr %{{.*}})
// IR:       omp.inner.for.body:
// IR:       br i1 %{{.*}}, label %omp.body.next, label %omp.body.continue
// IR:       call void @__kmpc_doacross_fini(
extern "C" void ordered_collapse2_tile1d(int n) {
#pragma omp for collapse(2) ordered(2)
#pragma omp tile sizes(5)
  for (int i = 0; i < n; ++i)
    body(i);
}

#endif /* HEADER */
