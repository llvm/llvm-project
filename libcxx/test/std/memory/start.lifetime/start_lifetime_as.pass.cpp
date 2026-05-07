//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>
// template<class T> T* start_lifetime_as(void* p) noexcept;
// template<class T> const T* start_lifetime_as(const void* p) noexcept;
// template<class T> volatile T* start_lifetime_as(volatile void* p) noexcept;
// template<class T> const volatile T* start_lifetime_as(const volatile void* p) noexcept;

#include <memory>
#include <cassert>
#include <type_traits>

struct Trivial {
  int x;
  float y;
};

struct NestedArray {
  int arr[3][4];
};

static_assert(noexcept(std::start_lifetime_as<int>(std::declval<void*>())));
static_assert(noexcept(std::start_lifetime_as<int>(std::declval<const void*>())));
static_assert(noexcept(std::start_lifetime_as<int>(std::declval<volatile void*>())));
static_assert(noexcept(std::start_lifetime_as<int>(std::declval<const volatile void*>())));

static_assert(std::is_same_v<decltype(std::start_lifetime_as<int>(std::declval<void*>())), int*>);
static_assert(std::is_same_v<decltype(std::start_lifetime_as<int>(std::declval<const void*>())), const int*>);
static_assert(std::is_same_v<decltype(std::start_lifetime_as<int>(std::declval<volatile void*>())), volatile int*>);
static_assert(
    std::is_same_v<decltype(std::start_lifetime_as<int>(std::declval<const volatile void*>())), const volatile int*>);

void test_cv_qualifiers() {
  alignas(Trivial) unsigned char buffer[sizeof(Trivial)];
  void* p                  = buffer;
  const void* cp           = buffer;
  volatile void* vp        = buffer;
  const volatile void* cvp = buffer;

  Trivial* ptr                  = std::start_lifetime_as<Trivial>(p);
  const Trivial* cptr           = std::start_lifetime_as<Trivial>(cp);
  volatile Trivial* vptr        = std::start_lifetime_as<Trivial>(vp);
  const volatile Trivial* cvptr = std::start_lifetime_as<Trivial>(cvp);

  assert(static_cast<void*>(ptr) == buffer);
  assert(static_cast<const void*>(cptr) == buffer);
  assert(static_cast<volatile void*>(vptr) == buffer);
  assert(static_cast<const volatile void*>(cvptr) == buffer);
}

void test_multi_dimensional_arrays() {
  alignas(NestedArray) unsigned char buffer[sizeof(NestedArray)];

  // P2679R2: The builtin should unwrap NestedArray completely
  NestedArray* ptr = std::start_lifetime_as<NestedArray>(buffer);
  assert(static_cast<void*>(ptr) == buffer);

  using Matrix = double[4][4];
  alignas(double) unsigned char mat_buffer[sizeof(Matrix)];
  Matrix* mat_ptr = std::start_lifetime_as<Matrix>(mat_buffer);
  assert(static_cast<void*>(mat_ptr) == mat_buffer);
}

int main(int, char**) {
  test_cv_qualifiers();
  test_multi_dimensional_arrays();
  return 0;
}
