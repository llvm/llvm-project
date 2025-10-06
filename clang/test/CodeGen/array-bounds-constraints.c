// This test verifies that clang generates llvm.assume statements to inform the
// optimizer that array subscripts are within bounds to enable better optimization.
// RUN: %clang_cc1 -emit-llvm -O2 -fassume-array-bounds %s -o - | FileCheck %s

// Verify no assumes are generated.
// RUN: %clang_cc1 -emit-llvm -O2 -fno-assume-array-bounds %s -o - | FileCheck %s -check-prefix=NO-FLAG

// CHECK-LABEL: define {{.*}} @test_simple_array
// NO-FLAG-LABEL: define {{.*}} @test_simple_array
void init_array(int *arr);
int test_simple_array(int i) {
  int arr[10];  // C arrays are 0-based: valid indices are [0, 9]
  init_array(arr);  // Initialize to avoid UB from uninitialized read.
  // CHECK: %{{.*}} = icmp ult i32 %i, 10
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  // NO-FLAG-NOT: call void @llvm.assume
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_multidimensional_array
int test_multidimensional_array(int i, int j) {
  int arr[5][8];  // Valid indices: i in [0, 4], j in [0, 7]
  init_array(arr[0]);  // Initialize to avoid UB from uninitialized read.
  // CHECK: %{{.*}} = icmp ult i32 %i, 5
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  // CHECK: %{{.*}} = icmp ult i32 %j, 8
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  return arr[i][j];
}

// CHECK-LABEL: define {{.*}} @test_unsigned_index
int test_unsigned_index(unsigned int i) {
  int arr[10];
  init_array(arr);  // Initialize to avoid UB from uninitialized read.
  // CHECK: %{{.*}} = icmp ult i32 %i, 10
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_store_undef
void test_store_undef(int i, int value) {
  int arr[10];
  // CHECK: %{{.*}} = icmp ult i32 %i, 10
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  arr[i] = value;
  init_array(arr);  // Avoid optimization of the above statement.
}
