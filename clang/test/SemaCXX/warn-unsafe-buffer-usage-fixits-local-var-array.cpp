// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
typedef int * Int_ptr_t;
typedef int Int_t;

void local_array(unsigned idx) {
  int buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:17}:"std::array<int, 10> buffer"
  buffer[idx] = 0;
}

void weird_whitespace_in_declaration(unsigned idx) {
  int      buffer_w   [       10 ] ;
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:35}:"std::array<int,        10 > buffer_w"
  buffer_w[idx] = 0;
}

void weird_comments_in_declaration(unsigned idx) {
  int   /* [ ] */   buffer_w  /* [ ] */ [ /* [ ] */ 10 /* [ ] */ ] ;
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:67}:"std::array<int   /* [ ] */,  /* [ ] */ 10 /* [ ] */ > buffer_w"
  buffer_w[idx] = 0;
}

void unsupported_multi_decl1(unsigned idx) {
  int a, buffer[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer[idx] = 0;
}

void unsupported_multi_decl2(unsigned idx) {
  int buffer[10], b;
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer[idx] = 0;
}

void local_array_ptr_to_const(unsigned idx, const int*& a) {
  const int * buffer[10] = {a};
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:25}:"std::array<const int *, 10> buffer"
  a = buffer[idx];
}

void local_array_const_ptr(unsigned idx, int*& a) {
  int * const buffer[10] = {a};
// FIXME: implement support
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  a = buffer[idx];

}

void local_array_const_ptr_to_const(unsigned idx, const int*& a) {
  const int * const buffer[10] = {a};
// FIXME: implement support
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  a = buffer[idx];

}

template<typename T>
void local_array_in_template(unsigned idx) {
  T buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:.*-[[@LINE-1]]:.*}
  buffer[idx] = 0;
}
// Instantiate the template function to force its analysis.
template void local_array_in_template<int>(unsigned); // FIXME: expected note {{in instantiation of}}

typedef unsigned int uint;
void typedef_as_elem_type(unsigned idx) {
  uint buffer[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:18}:"std::array<uint, 10> buffer"
  buffer[idx] = 0;
}

void macro_as_elem_type(unsigned idx) {
#define MY_INT int
  MY_INT buffer[10];
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
  // This will very likely be buffer overflow but hardened std::array catch these at runtime.
  buffer[signed_idx] = 0;
}
