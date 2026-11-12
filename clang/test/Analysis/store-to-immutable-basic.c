// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -verify %s

// Test basic functionality of StoreToImmutable checker for the C programming language.

const int tentative_global_const; // expected-note {{Memory region is declared as immutable here}}

void test_direct_write_to_tentative_const_global() {
  *(int*)&tentative_global_const = 100; // expected-warning {{Trying to write to immutable memory in global read-only storage}}
}

const int global_const = 42; // expected-note {{Memory region is declared as immutable here}}

void test_direct_write_to_const_global() {
  // This should trigger a warning about writing to immutable memory
  *(int*)&global_const = 100; // expected-warning {{Trying to write to immutable memory in global read-only storage}}
}

void test_write_through_const_pointer() {
  const int local_const = 10; // expected-note {{Memory region is declared as immutable here}}
  int *ptr = (int*)&local_const;
  *ptr = 20; // expected-warning {{Trying to write to immutable memory}}
}

void test_write_to_const_array() {
  const int arr[5] = {1, 2, 3, 4, 5}; // expected-note {{Enclosing memory region is declared as immutable here}}
  int *ptr = (int*)arr;
  ptr[0] = 10; // expected-warning {{Trying to write to immutable memory}}
}

struct TestStruct {
  const int x; // expected-note 2 {{Memory region is declared as immutable here}}
  int y;
};

void test_write_to_const_struct_member() {
  struct TestStruct s = {1, 2};
  int *ptr = (int*)&s.x;
  *ptr = 10; // expected-warning {{Trying to write to immutable memory}}
}

const int global_array[3] = {1, 2, 3}; // expected-note {{Enclosing memory region is declared as immutable here}}

void test_write_to_const_global_array() {
  int *ptr = (int*)global_array;
  ptr[0] = 10; // expected-warning {{Trying to write to immutable memory in global read-only storage}}
}

const struct TestStruct global_struct = {1, 2};

void test_write_to_const_global_struct() {
  int *ptr = (int*)&global_struct.x;
  *ptr = 10; // expected-warning {{Trying to write to immutable memory in global read-only storage}}
}


void test_write_to_const_param(const int param) { // expected-note {{Memory region is declared as immutable here}}
  *(int*)&param = 100; // expected-warning {{Trying to write to immutable memory}}
}

void test_write_to_const_ptr_param(const int *param) {
  *(int*)param = 100; // expected-warning {{Trying to write to immutable memory}}
}

void test_write_to_const_array_param(const int arr[5]) {
  *(int*)arr = 100; // expected-warning {{Trying to write to immutable memory}}
}

struct ParamStruct {
  const int z; // expected-note 2 {{Memory region is declared as immutable here}}
  int w;
};

void test_write_to_const_struct_param(const struct ParamStruct s) {
  *(int*)&s.z = 100; // expected-warning {{Trying to write to immutable memory}}
}

void test_write_to_const_struct_ptr_param(const struct ParamStruct *s) {
  *(int*)&s->z = 100; // expected-warning {{Trying to write to immutable memory}}
}

void test_write_to_nonconst() {
  int non_const = 42;
  *(int*)&non_const = 100; // No warning expected
}

int global_non_const = 42;

void test_write_to_nonconst_global() {
  *(int*)&global_non_const = 100; // No warning expected
}

struct NonConstStruct {
  int x;
  int y;
};

void test_write_to_nonconst_struct_member() {
  struct NonConstStruct s = {1, 2};
  *(int*)&s.x = 100; // No warning expected
}

void test_write_to_nonconst_param(int param) {
  *(int*)&param = 100; // No warning expected
}

void test_normal_assignment() {
  int x = 42;
  x = 100; // No warning expected
}

void test_const_ptr_to_nonconst_data() {
  int data = 42;
  const int *ptr = &data;
  *(int*)ptr = 100; // No warning expected
}

void test_const_ptr_to_const_data() {
  const int data = 42; // expected-note {{Memory region is declared as immutable here}}
  const int *ptr = &data;
  *(int*)ptr = 100; // expected-warning {{Trying to write to immutable memory}}
} 
