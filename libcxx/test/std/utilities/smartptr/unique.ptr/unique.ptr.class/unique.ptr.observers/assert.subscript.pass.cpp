//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <memory>
//
// unique_ptr<T[]>
//
// T& operator[](std::size_t);

// This test ensures that we catch an out-of-bounds access in std::unique_ptr<T[]>::operator[]
// when unique_ptr has the appropriate ABI configuration.

#include <memory>
#include <cstddef>
#include <string>

#include "check_assertion.h"
#include "type_algorithms.h"

struct MyDeleter {
  MyDeleter() = default;

  // required to exercise converting move-constructor
  template <class T>
  MyDeleter(std::default_delete<T> const&) {}

  // required to exercise converting move-assignment
  template <class T>
  MyDeleter& operator=(std::default_delete<T> const&) {
    return *this;
  }

  template <class T>
  void operator()(T* ptr) const {
    delete[] ptr;
  }
};

template <class WithCookie, class NoCookie>
void test() {
  // For types with an array cookie, we can always detect OOB accesses.
  {
    // Check with the default deleter
    {
      {
        std::unique_ptr<WithCookie[]> ptr(new WithCookie[5]);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
      {
        std::unique_ptr<WithCookie[]> ptr = std::make_unique<WithCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#if TEST_STD_VER >= 20
      {
        std::unique_ptr<WithCookie[]> ptr = std::make_unique_for_overwrite<WithCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = WithCookie(), "unique_ptr<T[]>::operator[](index): index out of range");
      }
#endif
    }

    // Check with a custom deleter
    {
      std::unique_ptr<WithCookie[], MyDeleter> ptr(new WithCookie[5]);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }
  }

  // For types that don't have an array cookie, things are a bit more complicated. We can detect OOB accesses
  // only when the unique_ptr is created via an API where the size is passed down to the library so that we
  // can store it inside the unique_ptr. That requires the appropriate ABI configuration to be enabled.
  //
  // Note that APIs that allow the size to be passed down to the library only support the default deleter
  // as of writing this test.
#if defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
  {
    {
      std::unique_ptr<NoCookie[]> ptr = std::make_unique<NoCookie[]>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }
#  if TEST_STD_VER >= 20
    {
      std::unique_ptr<NoCookie[]> ptr = std::make_unique_for_overwrite<NoCookie[]>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = NoCookie(), "unique_ptr<T[]>::operator[](index): index out of range");
    }
#  endif
  }
#endif

  // Make sure that we carry the bounds information properly through conversions, assignments, etc.
  // These tests are mostly relevant when the ABI setting is enabled (with a stateful bounds-checker),
  // but we still run them for types with an array cookie either way.
#if defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
  using Types = types::type_list<NoCookie, WithCookie>;
#else
  using Types = types::type_list<WithCookie>;
#endif
  types::for_each(Types(), []<class T> {
    // Bounds carried through move construction
    {
      std::unique_ptr<T[]> ptr = std::make_unique<T[]>(5);
      std::unique_ptr<T[]> other(std::move(ptr));
      TEST_LIBCPP_ASSERT_FAILURE(other[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }

    // Bounds carried through move assignment
    {
      std::unique_ptr<T[]> ptr = std::make_unique<T[]>(5);
      std::unique_ptr<T[]> other;
      other = std::move(ptr);
      TEST_LIBCPP_ASSERT_FAILURE(other[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }

    // Bounds carried through converting move-constructor
    {
      std::unique_ptr<T[]> ptr = std::make_unique<T[]>(5);
      std::unique_ptr<T[], MyDeleter> other(std::move(ptr));
      TEST_LIBCPP_ASSERT_FAILURE(other[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }

    // Bounds carried through converting move-assignment
    {
      std::unique_ptr<T[]> ptr = std::make_unique<T[]>(5);
      std::unique_ptr<T[], MyDeleter> other;
      other = std::move(ptr);
      TEST_LIBCPP_ASSERT_FAILURE(other[6], "unique_ptr<T[]>::operator[](index): index out of range");
    }
  });
}

template <std::size_t Size>
struct NoCookie {
  char padding[Size];
};

template <std::size_t Size>
struct WithCookie {
  WithCookie() = default;
  WithCookie(WithCookie const&) {}
  WithCookie& operator=(WithCookie const&) { return *this; }
  ~WithCookie() {}
  char padding[Size];
};

int main(int, char**) {
  test<WithCookie<1>, NoCookie<1>>();
  test<WithCookie<2>, NoCookie<2>>();
  test<WithCookie<3>, NoCookie<3>>();
  test<WithCookie<4>, NoCookie<4>>();
  test<WithCookie<8>, NoCookie<8>>();
  test<WithCookie<16>, NoCookie<16>>();
  test<WithCookie<32>, NoCookie<32>>();
  test<WithCookie<256>, NoCookie<256>>();
  test<std::string, int>();

  return 0;
}
