// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
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
  // CHECK: store i32 {{.*}}, ptr %i
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

void test_pointer_tile(void) {
  // CHECK-LABEL: define {{.*}} @test_pointer_tile
  int arr[20];
  int *p;
  #pragma omp tile sizes(5)
  for (p = arr; p < arr + 20; p++) {
    *p = 0;
  }
  // CHECK: for.end{{.*}}:
  // CHECK: [[START:%.*]] = load ptr, ptr %.capture_expr.
  // CHECK: [[END:%.*]] = load ptr, ptr %.capture_expr.{{[0-9]+}}
  // CHECK: [[STARTDUP:%.*]] = load ptr, ptr %.capture_expr.
  // CHECK: [[STARTCAST:%.*]] = ptrtoaddr ptr [[END]] to i64
  // CHECK: [[ENDCAST:%.*]] = ptrtoaddr ptr [[STARTDUP]] to i64
  // CHECK: [[DIFF:%.*]] = sub i64 [[STARTCAST]], [[ENDCAST]]
  // CHECK: [[NUMELEM:%.*]] = sdiv exact i64 [[DIFF]], 4
  // CHECK: [[SUB:%.*]] = sub nsw i64 [[NUMELEM]], 1
  // CHECK: [[ADD:%.*]] = add nsw i64 [[SUB]], 1
  // CHECK: [[DIV:%.*]] = sdiv i64 [[ADD]], 1
  // CHECK: [[OFFSET:%.*]] = mul nsw i64 [[DIV]], 1
  // CHECK: [[FINALPTR:%.*]] = getelementptr inbounds i32, ptr [[START]], i64 [[OFFSET]]
  // CHECK: store ptr [[FINALPTR]], ptr %p
  // After loop: p should point to arr + 20 (loop-exit value)
}

void test_pointer_reverse(void) {
  // CHECK-LABEL: define {{.*}} @test_pointer_reverse
  int arr[15];
  int *q;
  #pragma omp reverse
  for (q = arr; q < arr + 15; q++) {
    *q = 1;
  }
  // CHECK: for.end{{.*}}:
  // CHECK: load ptr, ptr %.capture_expr.
  // CHECK: load ptr, ptr %.capture_expr.{{[0-9]+}}
  // CHECK: ptrtoaddr ptr
  // CHECK: ptrtoaddr ptr
  // CHECK: sub i64
  // CHECK: sdiv exact i64
  // CHECK: mul nsw i64 {{.*}}, 1
  // CHECK: [[FINALPTR:%.*]] = getelementptr inbounds i32, ptr {{.*}}, i64
  // CHECK: store ptr [[FINALPTR]], ptr %q
  // After loop: q should point to arr + 15 (loop-exit value)
}

void test_pointer_step(void) {
  // CHECK-LABEL: define {{.*}} @test_pointer_step
  char buffer[30];
  char *ptr;
  #pragma omp tile sizes(4)
  for (ptr = buffer; ptr < buffer + 30; ptr += 2) {
    *ptr = 'x';
  }
  // CHECK: for.end{{.*}}:
  // CHECK: load ptr, ptr %.capture_expr.
  // CHECK: load ptr, ptr %.capture_expr.{{[0-9]+}}
  // CHECK: ptrtoaddr ptr
  // CHECK: ptrtoaddr ptr
  // CHECK: sub i64
  // CHECK: mul nsw i64 {{.*}}, 2
  // CHECK: [[FINALPTR:%.*]] = getelementptr inbounds i8, ptr {{.*}}, i64
  // CHECK: store ptr [[FINALPTR]], ptr %ptr
  // After loop: ptr should point to buffer + 30 (loop-exit value)
}

void test_pointer_fuse(void) {
  // CHECK-LABEL: define {{.*}} @test_pointer_fuse
  int arr1[10];
  int arr2[15];
  int *p;
  int *q;
  #pragma omp fuse looprange(1,2)
  {
    for (p = arr1; p < arr1 + 10; p++) {
      *p = 0;
    }
    for (q = arr2; q < arr2 + 15; q++) {
      *q = 1;
    }
  }
  // CHECK: for.end{{.*}}:
  // Finalization for p (first fused loop)
  // CHECK: load ptr, ptr %.capture_expr.
  // CHECK: ptrtoaddr ptr
  // CHECK: ptrtoaddr ptr
  // CHECK: sub i64
  // CHECK: sdiv exact i64
  // CHECK: mul nsw i64 {{.*}}, 1
  // CHECK: [[P_FINAL:%.*]] = getelementptr inbounds i32, ptr {{.*}}, i64
  // CHECK: store ptr [[P_FINAL]], ptr %p
  // Finalization for q (second fused loop)
  // CHECK: mul nsw i64 {{.*}}, 1
  // CHECK: [[Q_FINAL:%.*]] = getelementptr inbounds i32, ptr {{.*}}, i64
  // CHECK: store ptr [[Q_FINAL]], ptr %q
  // After fuse: p should point to arr1 + 10, q should point to arr2 + 15
}
