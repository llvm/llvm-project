// RUN: %clang_cc1 -triple amdgcn---amdgiz -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __attribute__((address_space(1))) void *global_ptr_t;
typedef __attribute__((address_space(2))) void *constant_ptr_t;
typedef __attribute__((address_space(3))) void *local_ptr_t;
typedef __attribute__((address_space(5))) void *private_ptr_t;

void test() {
  static_assert(sizeof(size_t) == 8, "bad size");
  static_assert(alignof(size_t) == 8, "bad alignment");
  static_assert(sizeof(intptr_t) == 8, "bad size");
  static_assert(alignof(intptr_t) == 8, "bad alignment");
  static_assert(sizeof(uintptr_t) == 8, "bad size");
  static_assert(alignof(uintptr_t) == 8, "bad alignment");
  static_assert(sizeof(ptrdiff_t) == 8, "bad size");
  static_assert(alignof(ptrdiff_t) == 8, "bad alignment");

  static_assert(sizeof(char) == 1, "bad size");
  static_assert(alignof(char) == 1, "bad alignment");
  static_assert(sizeof(short) == 2, "bad size");
  static_assert(alignof(short) == 2, "bad alignment");
  static_assert(sizeof(int) == 4, "bad size");
  static_assert(alignof(int) == 4, "bad alignment");
  static_assert(sizeof(long) == 8, "bad size");
  static_assert(alignof(long) == 8, "bad alignment");
  static_assert(sizeof(long long) == 8, "bad size");
  static_assert(alignof(long long) == 8, "bad alignment");
  static_assert(sizeof(float) == 4, "bad size");
  static_assert(alignof(float) == 4, "bad alignment");
  static_assert(sizeof(double) == 8, "bad size");
  static_assert(alignof(double) == 8, "bad alignment");

  static_assert(sizeof(void*) == 8, "bad size");
  static_assert(alignof(void*) == 8, "bad alignment");
  static_assert(sizeof(global_ptr_t) == 8, "bad size");
  static_assert(alignof(global_ptr_t) == 8, "bad alignment");
  static_assert(sizeof(constant_ptr_t) == 8, "bad size");
  static_assert(alignof(constant_ptr_t) == 8, "bad alignment");
  static_assert(sizeof(local_ptr_t) == 4, "bad size");
  static_assert(alignof(local_ptr_t) == 4, "bad alignment");
  static_assert(sizeof(private_ptr_t) == 4, "bad size");
  static_assert(alignof(private_ptr_t) == 4, "bad alignment");
}
