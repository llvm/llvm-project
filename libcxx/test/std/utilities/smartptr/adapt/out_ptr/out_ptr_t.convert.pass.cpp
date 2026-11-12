//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [out.ptr.t], class template out_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class out_ptr_t;                                          // since c++23

// operator Pointer*() const noexcept;
// operator void**() const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

int main(int, char**) {
  // operator Pointer*()
  {
    std::unique_ptr<int> uPtr;

    const std::out_ptr_t<std::unique_ptr<int>, int*> oPtr{uPtr};

    static_assert(noexcept(oPtr.operator int**()));
    std::same_as<int**> decltype(auto) pPtr = oPtr.operator int**();

    assert(*pPtr == nullptr);
  }

  {
    std::unique_ptr<int, std::default_delete<int>> uPtr;

    const std::out_ptr_t<decltype(uPtr), int*, std::default_delete<int>> oPtr{uPtr, std::default_delete<int>{}};

    static_assert(noexcept(oPtr.operator int**()));
    std::same_as<int**> decltype(auto) pPtr = oPtr.operator int**();

    assert(*pPtr == nullptr);
  }

  // operator void**()
  {
    std::unique_ptr<int> uPtr;

    const std::out_ptr_t<std::unique_ptr<int>, void*> oPtr{uPtr};

    static_assert(noexcept(oPtr.operator void**()));
    std::same_as<void**> decltype(auto) pPtr = oPtr.operator void**();

    assert(*pPtr == nullptr);
  }

  return 0;
}
