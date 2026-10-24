// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// C++ companion of loop_transform_iv.c: loop variables of loops affected by a
// loop-transformation construct are finalized to their loop-exit values
// (OpenMP 6.0), here for C++-only loop forms.

// A range-based for is a canonical loop; its iterator is restored after a
// fuse like any other loop variable.
extern "C" void test_fuse_range_for(void) {
  // CHECK-LABEL: define {{.*}} @test_fuse_range_for
  int arr[8];
  int t;
  int *q;
  #pragma omp fuse
  {
    for (t = 0; t < 4; ++t)
      ;
    for (q = arr; q != arr + 8; ++q)
      ;
    for (int &x : arr)
      ;
  }
  // CHECK: store i32 4, ptr %t
  // q = start + trip count, computed from the captured bounds.
  // CHECK: [[QEND:%.*]] = getelementptr inbounds i32, ptr %{{.*}}, i64 %{{.*}}
  // CHECK-NEXT: store ptr [[QEND]], ptr %q
  // CHECK: store ptr {{.*}}, ptr %__begin
}

// Control: tile restores the range-for iterator as well.
extern "C" void test_tile_range_for(void) {
  // CHECK-LABEL: define {{.*}} @test_tile_range_for
  int arr[8];
  #pragma omp tile sizes(2)
  for (int &x : arr)
    ;
  // CHECK: store ptr {{.*}}, ptr %__begin
}
