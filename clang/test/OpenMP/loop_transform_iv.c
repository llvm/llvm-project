// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Test that loop variables are correctly finalized after loop-transformation
// constructs as required by OpenMP 6.0 spec (pg 371, lines 19-21).

void test_tile(void) {
  // CHECK-LABEL: define {{.*}} @test_tile
  int i;
  #pragma omp tile sizes(2)
  for (i = 1; i <= 10; i++) {
  }
  // CHECK: store i32 11, ptr %i
  // After loop: i should be 11 (loop-exit value per OpenMP 6.0 spec)
}

void test_stripe(void) {
  // CHECK-LABEL: define {{.*}} @test_stripe
  int i;
  #pragma omp stripe sizes(3)
  for (i = 0; i < 10; i++) {
  }
  // CHECK: store i32 10, ptr %i
  // After loop: i should be 10 (loop-exit value per OpenMP 6.0 spec)
}

void test_tile_nested(void) {
  // CHECK-LABEL: define {{.*}} @test_tile_nested
  int i, j;
  #pragma omp tile sizes(2, 3)
  for (i = 0; i < 10; i++)
    for (j = 0; j < 5; j++) {
    }
  // CHECK: store i32 10, ptr %i
  // CHECK: store i32 5, ptr %j
  // After loop: i should be 10, j should be 5 (loop-exit values per OpenMP 6.0 spec)
}

void test_tile_stride(void) {
  // CHECK-LABEL: define {{.*}} @test_tile_stride
  int i;
  #pragma omp tile sizes(4)
  for (i = 5; i <= 15; i += 2) {
  }
  // CHECK: store i32 17, ptr %i
  // After loop: i should be 17 (loop-exit value per OpenMP 6.0 spec)
}

void test_tile_variable_bound(int n) {
  // CHECK-LABEL: define {{.*}} @test_tile_variable_bound
  int i;
  #pragma omp tile sizes(3)
  for (i = 0; i < n; i++) {
  }
  // CHECK: for.end14:
  // CHECK-NEXT: %[[NUMITER:.*]] = load i32, ptr %.capture_expr.
  // CHECK-NEXT: %[[SUB1:.*]] = sub nsw i32 %[[NUMITER]], 0
  // CHECK-NEXT: %[[DIV:.*]] = sdiv i32 %[[SUB1]], 1
  // CHECK-NEXT: %[[MUL:.*]] = mul nsw i32 %[[DIV]], 1
  // CHECK-NEXT: %[[FINAL:.*]] = add nsw i32 0, %[[MUL]]
  // CHECK-NEXT: store i32 %[[FINAL]], ptr %i
  // After loop: i should equal n (loop-exit value per OpenMP 6.0 spec)
}

void test_reverse(void) {
  // CHECK-LABEL: define {{.*}} @test_reverse
  int i;
  #pragma omp reverse
  for (i = 0; i < 10; i++) {
  }
  // CHECK: store i32 10, ptr %i
  // After loop: i should be 10 (loop-exit value per OpenMP 6.0 spec)
}

void test_interchange(void) {
  // CHECK-LABEL: define {{.*}} @test_interchange
  int i, j;
  #pragma omp interchange permutation(2, 1)
  for (i = 0; i < 10; i++)
    for (j = 0; j < 5; j++) {
    }
  // CHECK: store i32 10, ptr %i
  // CHECK: store i32 5, ptr %j
  // After loop: i should be 10, j should be 5 (loop-exit values per OpenMP 6.0 spec)
}

void test_for_workshare(void) {
  // CHECK-LABEL: define {{.*}} @test_for_workshare
  int i;
  #pragma omp for
  for (i = 5; i <= 10; i += 2) {
  }
  // CHECK: omp.loop.exit:
  // CHECK-NEXT: call void @__kmpc_for_static_fini
  // CHECK-NEXT: call void @__kmpc_barrier
  // CHECK-NEXT: ret void
  // omp for worksharing doesn't finalize the loop variable after the loop
}

void test_for_lastprivate(void) {
  // CHECK-LABEL: define {{.*}} @test_for_lastprivate
  int i;
  #pragma omp parallel
  #pragma omp for lastprivate(i)
  for (i = 5; i <= 10; i += 2) {
  }
  // CHECK: .omp.lastprivate.then:
  // CHECK-NEXT: store i32 11, ptr %i{{.*}}, align 4
  // CHECK-NEXT: %{{.*}} = load i32, ptr %i{{.*}}, align 4
  // CHECK-NEXT: store i32 %{{.*}}, ptr %{{.*}}, align 4
  // omp for with lastprivate DOES finalize the loop variable
  // i = last_iteration_value + stride = 9 + 2 = 11
}

void test_tile_workshare(void) {
  // CHECK-LABEL: define {{.*}} @test_tile_workshare
  int i;
  #pragma omp for
  #pragma omp tile sizes(2)
  for (i = 1; i <= 10; i++) {
  }
  // CHECK: omp.loop.exit:
  // CHECK-NOT: store {{.*}}, ptr %i
  // CHECK: call void @__kmpc_for_static_fini
  // i is private to the `for` construct; its post-loop value is
  // unspecified here (same as plain omp-for), so no restoring store
  // is expected for i after omp.loop.exit.
}

void test_tile_zero_tripcount(void) {
  // CHECK-LABEL: define {{.*}} @test_tile_zero_tripcount
  int i;
  #pragma omp tile sizes(2)
  for (i = 10; i < 5; i++) {
  }
  // CHECK: store i32 10, ptr %i
  // Loop doesn't execute (tripcount = 0), i should remain at initial value 10.
}

void test_stripe_negative_step(void) {
  // CHECK-LABEL: define {{.*}} @test_stripe_negative_step
  int i;
  #pragma omp stripe sizes(3)
  for (i = 10; i > 0; i--) {
  }
  // CHECK: store i32 0, ptr %i
  // Loop counts backward from 10 to 1, loop-exit value should be 0.
}
