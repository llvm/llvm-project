// This test verifies that clang generates llvm.assume statements to inform the
// optimizer that array subscripts are within bounds to enable better optimization.
// Use -O1 -disable-llvm-optzns to check assumes before DropUnnecessaryAssumesPass
// drops them (assumes with llvm.array.bounds metadata are dropped before vectorization).
// RUN: %clang_cc1 -emit-llvm -O1 -disable-llvm-optzns -fassume-array-bounds %s -o - | FileCheck %s

// Verify no assumes are generated.
// RUN: %clang_cc1 -emit-llvm -O1 -disable-llvm-optzns -fno-assume-array-bounds %s -o - | FileCheck %s -check-prefix=NO-FLAG

// CHECK-LABEL: define {{.*}} @test_simple_array
// NO-FLAG-LABEL: define {{.*}} @test_simple_array
void init_array(int *arr);
int test_simple_array(int i) {
  int arr[10];
  init_array(arr);
  // CHECK: call void @llvm.assume(i1 %{{.*}}){{.*}}!llvm.array.bounds
  // NO-FLAG-NOT: call void @llvm.assume
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_multidimensional_array
int test_multidimensional_array(int i, int j) {
  int arr[5][8];
  init_array(arr[0]);
  // CHECK: call void @llvm.assume(i1 %{{.*}}){{.*}}!llvm.array.bounds
  // CHECK: call void @llvm.assume(i1 %{{.*}}){{.*}}!llvm.array.bounds
  return arr[i][j];
}

// CHECK-LABEL: define {{.*}} @test_unsigned_index
int test_unsigned_index(unsigned int i) {
  int arr[10];
  init_array(arr);
  // CHECK: call void @llvm.assume(i1 %{{.*}}){{.*}}!llvm.array.bounds
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_store
void test_store(int i, int value) {
  int arr[10];
  // CHECK: call void @llvm.assume(i1 %{{.*}}){{.*}}!llvm.array.bounds
  arr[i] = value;
  init_array(arr);
}
