// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-static-sized-array \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -verify %s

// CHECK-NOT: [-Wunsafe-buffer-usage]
// expected-no-diagnostics

void foo(unsigned idx) {
  int buffer[10];
  buffer[idx] = 0;
}

int global_buffer[10];
void foo2(unsigned idx) { global_buffer[idx] = 0; }

struct Foo {
  int member_buffer[10];
  int x;
};

void foo2(Foo &f, unsigned idx) { f.member_buffer[idx] = 0; }

void constant_idx_safe(unsigned idx) {
  int buffer[10];
  buffer[9] = 0;
}

void constant_idx_safe0(unsigned idx) {
  int buffer[10];
  buffer[0] = 0;
}

int array[10];

void circular_access_unsigned(unsigned idx) {
  array[idx % 10];
  array[idx % 11];
  array[(idx + 3) % 10];
  array[(--idx) % 8];
  array[idx & 9 % 10];
  array[9 & idx % 11];
  array[12 % 10];
}

void circular_access_signed(int idx) { array[idx % 10]; }

void masked_idx1(unsigned long long idx, Foo f) {
  // Bitwise and operation
  array[idx & 5] = 10;
  array[5 & idx] = 12;
  array[idx & 11 & 5] = 3;
  array[idx & 11] = 20;
  array[idx &= 5];
  array[f.x & 5];
  array[5 & f.x];
  array[f.x & (-5)];
}

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

void type_conversions(uint64_t idx1, uint32_t idx2, uint8_t idx3) {
  array[(uint32_t)idx1 & 3];
  array[idx2 & 3];
  array[idx3 & 3];
}

int array2[5];

void masked_idx_safe(unsigned long long idx) {
  array2[6 & 5];
  array2[6 & idx & (idx + 1) & 5];
}

void constant_idx_unsafe(unsigned idx) {
  int buffer[10];
  buffer[10] = 0;
}

void constant_id_string(unsigned idx) {
  char safe_char = "abc"[1];
  safe_char = ""[0];
  safe_char = "\0"[0];

  char abcd[5] = "abc";
  abcd[2];

  char unsafe_char = "abc"[3];
  unsafe_char = "abc"[-1];
  unsafe_char = ""[1];
  unsafe_char = ""[idx];
}

typedef float Float4x4[4][4];

float two_dimension_array(Float4x4 &matrix, unsigned idx) {
  float a = matrix[0][4];

  a = matrix[0][3];

  a = matrix[4][0];

  a = matrix[idx][0];

  a = matrix[0][idx];

  a = matrix[idx][idx];

  return matrix[1][1];
}

typedef float Float2x3x4[2][3][4];
float multi_dimension_array(Float2x3x4 &matrix) {
  float *f = matrix[0][2];
  return matrix[1][2][3];
}

char array_strings[][11] = {"Apple", "Banana", "Cherry", "Date", "Elderberry"};

char array_string[] = "123456";

char access_strings() {
  char c = array_strings[0][4];
  c = array_strings[3][10];
  c = array_string[5];
  return c;
}

struct T {
  int array[10];
};

const int index = 1;

constexpr int get_const(int x) {
  if (x < 3)
    return ++x;
  else
    return x + 5;
};

void array_indexed_const_expr(unsigned idx) {
  int arr[10];
  arr[sizeof(int)] = 5;

  int array[sizeof(T)];
  array[sizeof(int)] = 5;
  array[sizeof(T) - 1] = 3;

  int k = arr[6 & 5];
  k = arr[2 << index];
  k = arr[8 << index];
  k = arr[16 >> 1];
  k = arr[get_const(index)];
  k = arr[get_const(5)];
  k = arr[get_const(4)];
}
