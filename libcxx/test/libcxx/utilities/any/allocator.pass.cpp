//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// Check that we're consistently using std::allocator_traits to
// allocate/deallocate/construct/destroy objects in std::any.
// See https://llvm.org/PR45099 for details.

#include <any>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "test_macros.h"


// Make sure we don't fit in std::any's SBO
struct Large { char big[sizeof(std::any) + 1]; };

// Make sure we fit in std::any's SBO
struct Small { };

bool Large_was_allocated = false;
bool Large_was_constructed = false;
bool Large_was_destroyed = false;
bool Large_was_deallocated = false;

bool Small_was_constructed = false;
bool Small_was_destroyed = false;

namespace std {
  template <>
  struct allocator<Large> {
    using value_type = Large;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    Large* allocate(std::size_t n) {
      Large_was_allocated = true;
      return static_cast<Large*>(::operator new(n * sizeof(Large)));
    }

    template <typename ...Args>
    void construct(Large* p, Args&& ...args) {
      new (p) Large(std::forward<Args>(args)...);
      Large_was_constructed = true;
    }

    void destroy(Large* p) {
      p->~Large();
      Large_was_destroyed = true;
    }

    void deallocate(Large* p, std::size_t) {
      Large_was_deallocated = true;
      return ::operator delete(p);
    }
  };

  template <>
  struct allocator<Small> {
    using value_type = Small;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    Small* allocate(std::size_t) { assert(false); return nullptr; }

    template <typename ...Args>
    void construct(Small* p, Args&& ...args) {
      new (p) Small(std::forward<Args>(args)...);
      Small_was_constructed = true;
    }

    void destroy(Small* p) {
      p->~Small();
      Small_was_destroyed = true;
    }

    void deallocate(Small*, std::size_t) { assert(false); }
  };
} // end namespace std


int main(int, char**) {
  // Test large types
  {
    {
      std::any a = Large();
      (void)a;

      assert(Large_was_allocated);
      assert(Large_was_constructed);
    }

    assert(Large_was_destroyed);
    assert(Large_was_deallocated);
  }

  // Test small types
  {
    {
      std::any a = Small();
      (void)a;

      assert(Small_was_constructed);
    }

    assert(Small_was_destroyed);
  }

  return 0;
}
