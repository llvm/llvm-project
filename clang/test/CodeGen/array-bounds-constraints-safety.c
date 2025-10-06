// RUN: %clang_cc1 -emit-llvm -O2 -fassume-array-bounds %s -o - | FileCheck %s
// Test that array bounds constraints are NOT applied to cases that might
// break real-world code with intentional out-of-bounds access patterns.

// C18 standard allows one-past-the-end pointers, and some legacy code
// intentionally accesses out-of-bounds for performance or compatibility.
// This test verifies that bounds constraints are only applied to safe cases.

// CHECK-LABEL: define {{.*}} @test_zero_length_array
struct ZeroLengthData {
    int count;
    int items[0];  // GNU C extension: zero-length array
};

int test_zero_length_array(struct ZeroLengthData *d, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Zero-length array as last field should not generate bounds constraints.
  // See https://gcc.gnu.org/onlinedocs/gcc/Zero-Length.html
  return d->items[i];
}

// CHECK-LABEL: define {{.*}} @test_flexible_array_member
struct Data {
    int count;
    int items[1];  // Flexible array member pattern (pre-C99 style)
};

int test_flexible_array_member(struct Data *d, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Flexible array member pattern (size 1 array as last field) should NOT
  // generate bounds constraints because items[1] is just a placeholder
  // for a larger array allocated with `malloc (sizeof (struct Data) + 42)`.
  return d->items[i];
}

// CHECK-LABEL: define {{.*}} @test_not_flexible_array
struct NotFlexible {
    int items[1];  // Size 1 array but NOT the last field.
    int count;     // Something comes after it.
};

int test_not_flexible_array(struct NotFlexible *s, int i) {
  // CHECK: call void @llvm.assume
  // This is NOT a flexible array pattern (not the last field),
  // so we're fine generating `assume(i < 1)`.
  return s->items[i];
}

// CHECK-LABEL: define {{.*}} @test_pointer_parameter
int test_pointer_parameter(int *arr, int i) {
  // CHECK-NOT: call void @llvm.assume
  // Pointer parameters should NOT generate bounds constraints
  // because we don't know the actual array size.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_vla
int test_vla(int n, int i) {
  int arr[n];  // Variable-length array.
  // CHECK-NOT: call void @llvm.assume
  // VLAs should NOT generate bounds constraints
  // because the size is dynamic.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_one_past_end
extern int extern_array[100];
int *test_one_past_end(void) {
  // CHECK-NOT: call void @llvm.assume
  // Taking address of one-past-the-end is allowed by C standard.
  // We should NOT assume anything about this access.
  return &extern_array[100];  // Legal: one past the end.
}

// CHECK-LABEL: define {{.*}} @test_extern_array
int test_extern_array(int i) {
  // CHECK: call void @llvm.assume
  // This will generate bounds constraints.
  // The array is a constant-size global array.
  // This is the safe case where we want optimization hints.
  return extern_array[i];
}

// CHECK-LABEL: define {{.*}} @test_local_constant_array
void init_array(int *arr);
int test_local_constant_array(int i) {
  int arr[10];
  init_array(arr);  // Initialize to avoid UB from uninitialized read.
  // CHECK: call void @llvm.assume
  // This will generate bounds constraints.
  // We know the exact size of this alloca array.
  // This is the safe case where we want optimization hints.
  return arr[i];
}

// CHECK-LABEL: define {{.*}} @test_malloc_array
int *my_malloc(int);
int test_malloc_array(int i) {
  // CHECK-NOT: call void @llvm.assume
  // Dynamically allocated arrays accessed via pointers do not get bounds
  // constraints.
  int *x = my_malloc(100 * sizeof(int));
  return x[i];
}
