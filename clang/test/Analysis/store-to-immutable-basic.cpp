// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -std=c++17 -verify %s

void test_write_to_const_ref_param(const int &param) {
  *(int*)&param = 100; // expected-warning {{Trying to write to immutable memory}}
}

// FIXME: This should warn in C mode too.
void test_write_to_string_literal() {
  char *str = (char*)"hello";
  str[0] = 'H'; // expected-warning {{Trying to write to immutable memory}}
}

struct ParamStruct {
  const int z; // expected-note {{Memory region is declared as immutable here}}
  int w;
};

void test_write_to_const_struct_ref_param(const ParamStruct &s) {
  *(int*)&s.z = 100; // expected-warning {{Trying to write to immutable memory}}
}

void test_const_ref_to_nonconst_data() {
  int data = 42;
  const int &ref = data;
  *(int*)&ref = 100; // No warning expected
} 

void test_const_ref_to_const_data() {
  const int data = 42; // expected-note {{Memory region is declared as immutable here}}
  const int &ref = data;
  *(int*)&ref = 100; // expected-warning {{Trying to write to immutable memory}}
} 

void test_ref_to_nonconst_data() {
  int data = 42;
  int &ref = data;
  ref = 100; // No warning expected
}

void test_ref_to_const_data() {
  const int data = 42; // expected-note {{Memory region is declared as immutable here}}
  int &ref = *(int*)&data;
  ref = 100; // expected-warning {{Trying to write to immutable memory}}
}

struct MultipleLayerStruct {
  MultipleLayerStruct();
  const int data; // expected-note {{Memory region is declared as immutable here}}
  const int buf[10]; // expected-note {{Enclosing memory region is declared as immutable here}}
};

MultipleLayerStruct MLS[10];

void test_multiple_layer_struct_array_member() {
  int *p = (int*)&MLS[2].data;
  *p = 4; // expected-warning {{Trying to write to immutable memory}}
}

void test_multiple_layer_struct_array_array_member() {
  int *p = (int*)&MLS[2].buf[3];
  *p = 4; // expected-warning {{Trying to write to immutable memory}}
}

struct StructWithNonConstMember {
  int x;
};

const StructWithNonConstMember SWNCM{0}; // expected-note {{Enclosing memory region is declared as immutable here}}

void test_write_to_non_const_member_of_const_struct() {
  *(int*)&SWNCM.x = 100; // expected-warning {{Trying to write to immutable memory in global read-only storage}}
}
