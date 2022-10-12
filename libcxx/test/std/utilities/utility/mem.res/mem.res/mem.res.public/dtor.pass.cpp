//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

//------------------------------------------------------------------------------
// TESTING virtual ~memory_resource()
//
// Concerns:
//  A) 'memory_resource' is destructible.
//  B) The destructor is implicitly marked noexcept.
//  C) The destructor is marked virtual.

#include <memory_resource>
#include <cassert>
#include <type_traits>

#include "test_std_memory_resource.h"

int main(int, char**) {
  static_assert(std::has_virtual_destructor_v<std::pmr::memory_resource>);
  static_assert(std::is_nothrow_destructible_v<std::pmr::memory_resource>);
  static_assert(std::is_abstract_v<std::pmr::memory_resource>);

  // Check that the destructor of `TestResource` is called when
  // it is deleted as a pointer to `memory_resource`.
  {
    using TR                     = TestResource;
    std::pmr::memory_resource* M = new TR(42);
    assert(TR::resource_alive == 1);
    assert(TR::resource_constructed == 1);
    assert(TR::resource_destructed == 0);

    delete M;

    assert(TR::resource_alive == 0);
    assert(TR::resource_constructed == 1);
    assert(TR::resource_destructed == 1);
  }

  return 0;
}
