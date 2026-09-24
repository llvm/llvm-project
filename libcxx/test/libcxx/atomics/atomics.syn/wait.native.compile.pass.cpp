//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads

// When __has_native_atomic_wait<T> is true, the atomic object's address will be directly passed to the platform's wait.
// This test ensures that types that do not satisfy the platform's wait requirement should have __has_native_atomic_wait<T> be false.

#include <atomic>
#include <cstddef>
#include <type_traits>

template <std::size_t Size, std::size_t Align>
struct alignas(Align) Data {
  char buffer[Size];
};

static_assert(std::__has_native_atomic_wait<std::__cxx_contention_t>);

#if defined(_LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE) && defined(__APPLE__)

static_assert(std::__has_native_atomic_wait<Data<4, 4>>);
static_assert(std::__has_native_atomic_wait<Data<8, 8>>);

static_assert(!std::has_unique_object_representations_v<Data<4, 8>>);
static_assert(!std::__has_native_atomic_wait<Data<4, 8>>,
              "Object with !has_unique_object_representations_v should not have native wait");

static_assert(!std::__has_native_atomic_wait<Data<1, 1>>, "Should only support native wait for 4 and 8 byte types");

// `__ulock_wait` requires the address is aligned to the requested size (4 or 8)

static_assert(!std::__has_native_atomic_wait<Data<4, 1>>,
              "Should only support native wait for types with properly aligned types");

static_assert(!std::__has_native_atomic_wait<Data<8, 1>>,
              "Should only support native wait for types with properly aligned types");

#endif
