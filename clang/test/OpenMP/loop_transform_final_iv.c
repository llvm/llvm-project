// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -emit-llvm %s -o -\
// RUN: | FileCheck %s

// expected-no-diagnostics

// LABEL: define {{.*}} void @test_tile
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: store i32 11, ptr [[I]], align 4
void test_tile() {
  int i;
#pragma omp tile sizes(2)
  for (i = 1; i <= 10; i++)
    {}
}

// LABEL: define {{.*}} void @test_tile_variable_bound(i32 noundef [[N:%.*]])
// CHECK: for.end14:
// CHECK-NEXT: load i32, ptr %.capture_expr., align 4
// CHECK-NEXT: sub nsw i32 {{.*}}, 0
// CHECK-NEXT: sdiv i32 {{.*}}, 1
// CHECK-NEXT: mul nsw i32 {{.*}}, 1
// CHECK-NEXT: add nsw i32 1, {{.*}}
// CHECK-NEXT: store i32 {{.*}}, ptr {{.*}}, align 4
void test_tile_variable_bound(int n) {
  int i;
#pragma omp tile sizes(3)
  for (i = 1; i <= n; i++)
    {}
}

// LABEL: define {{.*}} void @test_tile_nested
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: [[J:%.*]] = alloca i32, align 4
// CHECK: store i32 10, ptr [[I]], align 4
// CHECK: store i32 5, ptr [[J]], align 4
void test_tile_nested(void) {
  int i, j;
#pragma omp tile sizes(2,3)
  for (i = 0; i < 10; i++)
    for (j = 0; j < 5; j++)
      {}
}

// LABEL: define {{.*}} void @test_stripe
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: store i32 12, ptr [[I]], align 4
void test_stripe() {
  int i;
#pragma omp stripe sizes(3)
  for (i = 1; i <= 11; i++)
    {}
}

// LABEL: define {{.*}} void @test_tile_stride
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: store i32 17, ptr [[I]], align 4
void test_tile_stride() {
  int i;
#pragma omp tile sizes(4)
  for (i = 5; i <= 15; i += 2)
    {}
}

// LABEL: define {{.*}} void @test_reverse
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: store i32 10, ptr [[I]], align 4
void test_reverse() {
  int i;
#pragma omp reverse
  for(i = 0; i < 10; i++)
    {}
}

// LABEL: define {{.*}} void @test_interchange
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: [[J:%.*]] = alloca i32, align 4
// CHECK: store i32 10, ptr [[I]], align 4
// CHECK: store i32 5, ptr [[J]], align 4
void test_interchange() {
  int i, j;
#pragma omp interchange permutation(2, 1)
  for (i = 0; i < 10; i++)
    for (j = 0; j < 5; j++)
      {}
}

// LABEL: define {{.*}} void @test_fuse() #0 {
// CHECK: [[I:%.*]] = alloca i32, align 4
// CHECK: [[J:%.*]] = alloca i32, align 4
// CHECK: store i32 11, ptr [[I]], align 4
// CHECK: store i32 6, ptr [[J]], align 4
void test_fuse() {
  int i, j;
#pragma omp fuse
  {
    for (i = 0; i <= 10; i++)
      {}
    for (j = 0; j <= 5; j++)
      {}
  }
}

// LABEL: define {{.*}} void @test_for_only
// CHECK: omp.loop.exit:
// CHECK-NEXT: call void @__kmpc_for_static_fini
// CHECK-NEXT: call void @__kmpc_barrier
// CHECK-NEXT: ret void
void test_for_only() {
  int i;
#pragma omp for
  for (i = 5; i <= 10; i += 2)
    {}
}

// LABEL: define {{.*}} void @test_for_lastprivate.omp_outlined
// CHECK: .omp.lastprivate.then:
// CHECK-NEXT: store i32 11, ptr {{.*}}, align 4
// CHECK-NEXT: load i32, ptr {{.*}}, align 4
// CHECK-NEXT: store i32 {{.*}}, ptr {{.*}}, align 4
void test_for_lastprivate() {
  int i;
#pragma omp parallel
#pragma omp for lastprivate(i)
  for (i = 5; i <= 10; i += 2)
    {}
}

