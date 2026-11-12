// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
typedef int * Int_ptr_t;
typedef int Int_t;

void simple(unsigned idx) {
  int buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 10> buffer"
  buffer[idx] = 0;
}

void array2d(unsigned idx) {
  int buffer[10][10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer[idx][idx] = 0;
}

void array2d_vla(unsigned sz, unsigned idx) {
  int buffer1[10][sz];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  int buffer2[sz][10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer1[idx][idx] = 0;
  buffer2[idx][idx] = 0;
}

void array2d_assign_from_elem(unsigned idx) {
  int buffer[10][10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  int a = buffer[idx][idx];
}

void array2d_use(int *);
void array2d_call(unsigned idx) {
  int buffer[10][10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  array2d_use(buffer[idx]);
}
void array2d_call_vla(unsigned sz, unsigned idx) {
  int buffer[10][sz];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  array2d_use(buffer[idx]);
}

void array2d_typedef(unsigned idx) {
  typedef int ten_ints_t[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  ten_ints_t buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer[idx][idx] = 0;
}

void whitespace_in_declaration(unsigned idx) {
  int      buffer_w   [       10 ];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:35}:"std::array<int, 10> buffer_w"
  buffer_w[idx] = 0;
}

void comments_in_declaration(unsigned idx) {
  int   /* [A] */   buffer_w  /* [B] */ [  /* [C] */ 10 /* [D] */  ] ;
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:69}:"std::array<int   /* [A] */, /* [C] */ 10 /* [D] */> buffer_w"
  buffer_w[idx] = 0;
}

void initializer(unsigned idx) {
  int buffer[3] = {0};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:16}:"std::array<int, 3> buffer"

  int buffer2[3] = {0, 1, 2};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 3> buffer2"

  buffer[idx] = 0;
  buffer2[idx] = 0;
}

void auto_size(unsigned idx) {
  int buffer[] = {0, 1, 2};
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
// FIXME: implement support

  buffer[idx] = 0;
}

void universal_initialization(unsigned idx) {
  int buffer[] {0, 1, 2};
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
// FIXME: implement support

  buffer[idx] = 0;
}

void multi_decl1(unsigned idx) {
  int a, buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
// FIXME: implement support

  buffer[idx] = 0;
}

void multi_decl2(unsigned idx) {
  int buffer[10], b;
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
// FIXME: implement support

  buffer[idx] = 0;
}

void local_array_ptr_to_const(unsigned idx, const int*& a) {
  const int * buffer[10] = {a};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:25}:"std::array<const int *, 10> buffer"
  a = buffer[idx];
}

void local_array_const_ptr(unsigned idx, int*& a) {
  int * const buffer[10] = {a};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:25}:"std::array<int * const, 10> buffer"

  a = buffer[idx];
}

void local_array_const_ptr_via_typedef(unsigned idx, int*& a) {
  typedef int * const my_const_ptr;
  my_const_ptr buffer[10] = {a};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:26}:"std::array<my_const_ptr, 10> buffer"

  a = buffer[idx];
}

void local_array_const_ptr_to_const(unsigned idx, const int*& a) {
  const int * const buffer[10] = {a};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:31}:"std::array<const int * const, 10> buffer"

  a = buffer[idx];

}

template<typename T>
void unsupported_local_array_in_template(unsigned idx) {
  T buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  buffer[idx] = 0;
}
// Instantiate the template function to force its analysis.
template void unsupported_local_array_in_template<int>(unsigned);

typedef unsigned int my_uint;
void typedef_as_elem_type(unsigned idx) {
  my_uint buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:21}:"std::array<my_uint, 10> buffer"
  buffer[idx] = 0;
}

void decltype_as_elem_type(unsigned idx) {
  int a;
  decltype(a) buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:25}:"std::array<decltype(a), 10> buffer"
  buffer[idx] = 0;
}

void macro_as_elem_type(unsigned idx) {
#define MY_INT int
  MY_INT buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
// FIXME: implement support

  buffer[idx] = 0;
#undef MY_INT
}

void macro_as_identifier(unsigned idx) {
#define MY_BUFFER buffer
  int MY_BUFFER[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:20}:"std::array<int, 10> MY_BUFFER"
  MY_BUFFER[idx] = 0;
#undef MY_BUFFER
}

void macro_as_size(unsigned idx) {
#define MY_TEN 10
  int buffer[MY_TEN];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:21}:"std::array<int, MY_TEN> buffer"
  buffer[idx] = 0;
#undef MY_TEN
}

typedef unsigned int my_array[42];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
void typedef_as_array_type(unsigned idx) {
  my_array buffer;
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  buffer[idx] = 0;
}

void decltype_as_array_type(unsigned idx) {
  int buffer[42];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  decltype(buffer) buffer2;
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  buffer2[idx] = 0;
}

void constant_as_size(unsigned idx) {
  const unsigned my_const = 10;
  int buffer[my_const];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:23}:"std::array<int, my_const> buffer"
  buffer[idx] = 0;
}

void subscript_negative() {
  int buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 10> buffer"

  // For constant-size arrays any negative index will lead to buffer underflow.
  // std::array::operator[] has unsigned parameter so the value will be casted to unsigned.
  // This will very likely be buffer overflow but hardened std::array catch these at runtime.
  buffer[-5] = 0;
}

void subscript_signed(int signed_idx) {
  int buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 10> buffer"

  // For constant-size arrays any negative index will lead to buffer underflow.
  // std::array::operator[] has unsigned parameter so the value will be casted to unsigned.
  // This will very likely be buffer overflow but hardened std::array catches these at runtime.
  buffer[signed_idx] = 0;
}
