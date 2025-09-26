// Test that array bounds constraints generate llvm.assume statements for optimization hints.
// RUN: %clang_cc1 -emit-llvm -O2 %s -o - | FileCheck %s

// This test verifies that clang generates llvm.assume statements to inform the
// optimizer that array subscripts are within bounds to enable better optimization.

// CHECK-LABEL: define {{.*}} @test_simple_array
int test_simple_array(int i) {
  int arr[10];  // C arrays are 0-based: valid indices are [0, 9]
  // CHECK: %{{.*}} = icmp ult i32 %i, 10
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_multidimensional_array
int test_multidimensional_array(int i, int j) {
  int arr[5][8];  // Valid indices: i in [0, 4], j in [0, 7]
  // CHECK: %{{.*}} = icmp ult i32 %i, 5
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  // CHECK: %{{.*}} = icmp ult i32 %j, 8
  // CHECK: call void @llvm.assume(i1 %{{.*}})
  return arr[i][j];
}

// CHECK-LABEL: define {{.*}} @test_unsigned_index
int test_unsigned_index(unsigned int i) {
  int arr[10];
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
}
