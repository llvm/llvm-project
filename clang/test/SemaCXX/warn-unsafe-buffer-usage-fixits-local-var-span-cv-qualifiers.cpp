// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

int ptr(unsigned idx) {
  int * ptr = new int[1];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int a;
  a = ptr[idx];
  return a;
}

int ptr_to_const(unsigned idx) {
  const int * ptr = new int[1];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const>"
  int a;
  a = ptr[idx];
  return a;
}

int const_ptr(unsigned idx) {
  int * const ptr = new int[1];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int a;
  a = ptr[idx];
  return a;
}

int const_ptr_to_const(unsigned idx) {
  const int * const ptr = new int[1];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:14}:"std::span<int const>"
  int a;
  a = ptr[idx];
  return a;
}

int ptr_to_const_volatile(unsigned idx) {
  const volatile int * ptr = new int[1];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:23}:"std::span<int const volatile>"
  int a;
  a = ptr[idx];
  return a;
}

int const_volatile_ptr(unsigned idx) {
  int * const volatile ptr = new int[1];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int a;
  a = ptr[idx];
  return a;
}

int const_volatile_ptr_to_const_volatile(unsigned idx) {
  const volatile int * const volatile ptr = new int[1];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:23}:"std::span<int const volatile>"
  int a;
  a = ptr[idx];
  return a;
}

typedef const int * my_const_int_star;
int typedef_ptr_to_const(unsigned idx) {
  my_const_int_star ptr = new int[1];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int a;
  a = ptr[idx];
  return a;
}

typedef int * const my_int_star_const;
int typedef_const_ptr(unsigned idx) {
  my_int_star_const ptr = new int[1];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int a;
  a = ptr[idx];
  return a;
}

typedef const int * const my_const_int_star_const;
int typedef_const_ptr_to_const(unsigned idx) {
  my_const_int_star_const ptr = new int[1];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int a;
  a = ptr[idx];
  return a;
}

int ptr_to_decltype(unsigned idx) {
  int a;
  decltype(a) * ptr = new int[1];
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:16}:"std::span<decltype(a)>"
  a = ptr[idx];
  return a;
}

int decltype_ptr(unsigned idx) {
  int * p;
  decltype(p) ptr = new int[1];
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int a;
  a = ptr[idx];
  return a;
}
