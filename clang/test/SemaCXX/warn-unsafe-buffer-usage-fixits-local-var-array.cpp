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

void macro_as_identifier(unsigned idx) {
#define MY_BUFFER buffer
  // Although fix-its include macros, the macros do not overlap with
  // the bounds of the source range of these fix-its. So these fix-its
  // are valid.

  int MY_BUFFER[10];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:20}:"std::array<int, 10> MY_BUFFER"
  MY_BUFFER[idx] = 0;

#undef MY_BUFFER
}

void unsupported_fixit_overlapping_macro(unsigned idx) {
#define MY_INT int
  MY_INT buffer[10];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  buffer[idx] = 0;
#undef MY_INT
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
