// Use -O1 -disable-llvm-optzns to check assumes before DropUnnecessaryAssumesPass
// drops them (assumes with llvm.array.bounds metadata are dropped before vectorization).
// RUN: %clang_cc1 -emit-llvm -O1 -disable-llvm-optzns -fassume-array-bounds %s -o - | FileCheck %s

// Test that array bounds constraints are NOT applied to cases that might
// break real-world code with intentional out-of-bounds access patterns.

// CHECK-LABEL: define {{.*}} @test_zero_length_array
struct ZeroLengthData {
    int count;
    int items[0];  // GNU C extension: zero-length array
};

int test_zero_length_array(struct ZeroLengthData *d, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Zero-length array as last field should not generate bounds constraints.
  return d->items[i];
}

// CHECK-LABEL: define {{.*}} @test_flexible_array_member
struct Data {
    int count;
    int items[1];  // Flexible array member pattern (pre-C99 style)
};

int test_flexible_array_member(struct Data *d, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Flexible array member pattern should NOT generate bounds constraints.
  return d->items[i];
}

// CHECK-LABEL: define {{.*}} @test_not_flexible_array
struct NotFlexible {
    int items[1];  // Size 1 array but NOT the last field.
    int count;     // Something comes after it.
};

int test_not_flexible_array(struct NotFlexible *s, int i) {
  // CHECK: call void @llvm.assume{{.*}}!llvm.array.bounds
  // This is NOT a flexible array pattern, so we generate assume.
  return s->items[i];
}

// CHECK-LABEL: define {{.*}} @test_pointer_parameter
int test_pointer_parameter(int *arr, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Pointer parameters should NOT generate bounds constraints.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_vla
void init_vla(int *arr, int n);

int test_vla(int n, int i) {
  int arr[n];
  init_vla(arr, n);
  // CHECK: call void @llvm.assume{{.*}}!llvm.array.bounds
  // For VLAs, generate bounds constraints using the runtime size.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_one_past_end
extern int extern_array[100];
int *test_one_past_end(void) {
  // One-past-the-end address is legal per C standard.
  // We generate assume(i1 true) because 100 <= 100 is trivially true.
  // CHECK: call void @llvm.assume(i1 true){{.*}}!llvm.array.bounds
  return &extern_array[100];
}

// CHECK-LABEL: define {{.*}} @test_extern_array
int test_extern_array(int i) {
  // CHECK: call void @llvm.assume{{.*}}!llvm.array.bounds
  // Constant-size global array generates bounds constraints.
  return extern_array[i];
}

// CHECK-LABEL: define {{.*}} @test_local_constant_array
void init_array(int *arr);
int test_local_constant_array(int i) {
  int arr[10];
  init_array(arr);
  // CHECK: call void @llvm.assume{{.*}}!llvm.array.bounds
  // Local constant-size array generates bounds constraints.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_malloc_array
int *my_malloc(int);
int test_malloc_array(int i) {
  // CHECK-NOT: call void @llvm.assume
  // Dynamically allocated arrays via pointers do not get bounds constraints.
  int *x = my_malloc(100 * sizeof(int));
  return x[i];
}
