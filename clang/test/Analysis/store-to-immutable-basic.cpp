// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -verify %s

// Test basic functionality of StoreToImmutable checker
// This tests direct writes to immutable regions without function modeling

// Direct write to a const global variable
const int global_const = 42; // expected-note {{Memory region is in immutable space}}

void test_direct_write_to_const_global() {
  // This should trigger a warning about writing to immutable memory
  *(int*)&global_const = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write through a pointer to const memory
void test_write_through_const_pointer() {
  const int local_const = 10; // expected-note {{Memory region is in immutable space}}
  int *ptr = (int*)&local_const;
  *ptr = 20; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to string literal (should be in immutable space)
void test_write_to_string_literal() {
  char *str = (char*)"hello";
  str[0] = 'H'; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const array
void test_write_to_const_array() {
  const int arr[5] = {1, 2, 3, 4, 5};
  int *ptr = (int*)arr;
  ptr[0] = 10; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const struct member
struct TestStruct {
  const int x; // expected-note 2{{Memory region is in immutable space}}
  int y;
};

void test_write_to_const_struct_member() {
  TestStruct s = {1, 2};
  int *ptr = (int*)&s.x;
  *ptr = 10; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const global array
const int global_array[3] = {1, 2, 3};

void test_write_to_const_global_array() {
  int *ptr = (int*)global_array;
  ptr[0] = 10; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const global struct
const TestStruct global_struct = {1, 2};

void test_write_to_const_global_struct() {
  int *ptr = (int*)&global_struct.x;
  *ptr = 10; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const parameter
void test_write_to_const_param(const int param) { // expected-note {{Memory region is in immutable space}}
  *(int*)&param = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const reference parameter
void test_write_to_const_ref_param(const int &param) {
  *(int*)&param = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const pointer parameter
void test_write_to_const_ptr_param(const int *param) {
  *(int*)param = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const array parameter
void test_write_to_const_array_param(const int arr[5]) {
  *(int*)arr = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const struct parameter
struct ParamStruct {
  const int z; // expected-note 3{{Memory region is in immutable space}}
  int w;
};

void test_write_to_const_struct_param(const ParamStruct s) {
  *(int*)&s.z = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const struct reference parameter
void test_write_to_const_struct_ref_param(const ParamStruct &s) {
  *(int*)&s.z = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

// Write to const struct pointer parameter
void test_write_to_const_struct_ptr_param(const ParamStruct *s) {
  *(int*)&s->z = 100; // expected-warning {{Writing to immutable memory is undefined behavior}}
  // expected-note@-1 {{Writing to immutable memory is undefined behavior. This memory region is marked as immutable and should not be modified}}
}

//===--- NEGATIVE TEST CASES ---===//
// These tests should NOT trigger warnings

// Write to non-const variable (should not warn)
void test_write_to_nonconst() {
  int non_const = 42;
  *(int*)&non_const = 100; // No warning expected
}

// Write to non-const global variable (should not warn)
int global_non_const = 42;

void test_write_to_nonconst_global() {
  *(int*)&global_non_const = 100; // No warning expected
}

// Write to non-const struct member (should not warn)
struct NonConstStruct {
  int x;
  int y;
};

void test_write_to_nonconst_struct_member() {
  NonConstStruct s = {1, 2};
  *(int*)&s.x = 100; // No warning expected
}

// Write to non-const parameter (should not warn)
void test_write_to_nonconst_param(int param) {
  *(int*)&param = 100; // No warning expected
}

// Normal assignment to non-const variable (should not warn)
void test_normal_assignment() {
  int x = 42;
  x = 100; // No warning expected
}

// Write to non-const data through const pointer (should not warn - underlying memory is non-const)
void test_const_ptr_to_nonconst_data() {
  int data = 42;
  const int *ptr = &data;
  *(int*)ptr = 100; // No warning expected
}

// Write to non-const data through const reference (should not warn - underlying memory is non-const)
void test_const_ref_to_nonconst_data() {
  int data = 42;
  const int &ref = data;
  *(int*)&ref = 100; // No warning expected
} 