//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [inout.ptr.t], class template inout_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class inout_ptr_t;                                        // since c++23

// operator Pointer*() const noexcept;
// operator void**() const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

int main(int, char**) {
  {
    std::unique_ptr<int> uPtr = std::make_unique<int>(84);

    const std::inout_ptr_t<std::unique_ptr<int>, int*> ioPtr{uPtr};

    static_assert(noexcept(ioPtr.operator int**()));
    std::same_as<int**> decltype(auto) pPtr = ioPtr.operator int**();

    assert(**pPtr == 84);
  }
  {
    std::unique_ptr<int> uPtr = std::make_unique<int>(84);

    const std::inout_ptr_t<std::unique_ptr<int>, void*> ioPtr{uPtr};

    static_assert(noexcept(ioPtr.operator void**()));
    std::same_as<void**> decltype(auto) pPtr = ioPtr.operator void**();

    assert(**reinterpret_cast<int**>(pPtr) == 84);
  }

  return 0;
}
