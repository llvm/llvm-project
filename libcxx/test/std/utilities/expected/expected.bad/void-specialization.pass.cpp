//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<>
// class bad_expected_access<void> : public exception {
// protected:
//   bad_expected_access() noexcept;
//   bad_expected_access(const bad_expected_access&) noexcept;
//   bad_expected_access(bad_expected_access&&) noexcept;
//   bad_expected_access& operator=(const bad_expected_access&) noexcept;
//   bad_expected_access& operator=(bad_expected_access&&) noexcept;
//   ~bad_expected_access();
//
// public:
//   const char* what() const noexcept override;
// };

#include <cassert>
#include <exception>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct Inherit : std::bad_expected_access<void> {};

int main(int, char**) {
  // base class
  static_assert(std::is_base_of_v<std::exception, std::bad_expected_access<void>>);

  // default constructor
  {
    Inherit exc;
    ASSERT_NOEXCEPT(Inherit());
  }

  // copy constructor
  {
    Inherit exc;
    Inherit copy(exc);
    ASSERT_NOEXCEPT(Inherit(exc));
  }

  // move constructor
  {
    Inherit exc;
    Inherit copy(std::move(exc));
    ASSERT_NOEXCEPT(Inherit(std::move(exc)));
  }

  // copy assignment
  {
    Inherit exc;
    Inherit copy;
    [[maybe_unused]] Inherit& result = (copy = exc);
    ASSERT_NOEXCEPT(copy = exc);
  }

  // move assignment
  {
    Inherit exc;
    Inherit copy;
    [[maybe_unused]] Inherit& result = (copy = std::move(exc));
    ASSERT_NOEXCEPT(copy = std::move(exc));
  }

  // what()
  {
    Inherit exc;
    char const* what = exc.what();
    assert(what != nullptr);
    ASSERT_NOEXCEPT(exc.what());
  }

  return 0;
}
