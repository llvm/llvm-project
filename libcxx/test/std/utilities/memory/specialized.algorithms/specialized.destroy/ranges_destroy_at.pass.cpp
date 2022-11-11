//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>
//
// namespace ranges {
//   template<destructible T>
//     constexpr void destroy_at(T* location) noexcept; // since C++20
// }

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_macros.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::destroy_at)>);

struct NotNothrowDtrable {
  ~NotNothrowDtrable() noexcept(false) {}
};
static_assert(!std::is_invocable_v<decltype(std::ranges::destroy_at), NotNothrowDtrable*>);

struct Counted {
  int& count;

  constexpr Counted(int& count_ref) : count(count_ref) { ++count; }
  constexpr ~Counted() { --count; }

  friend void operator&(Counted) = delete;
};

struct VirtualCountedBase {
  int& count;

  constexpr VirtualCountedBase(int& count_ref) : count(count_ref) { ++count; }
  constexpr virtual ~VirtualCountedBase() { --count; }

  void operator&() const = delete;
};

struct VirtualCountedDerived : VirtualCountedBase {
  constexpr VirtualCountedDerived(int& count_ref) : VirtualCountedBase(count_ref) {}

  // Without a definition, GCC gives an error when the destructor is invoked in a constexpr context (see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93413).
  constexpr ~VirtualCountedDerived() override {}
};

constexpr bool test() {
  // Destroying a "trivial" object.
  {
    std::allocator<Counted> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Counted* buffer = Traits::allocate(alloc, 2);
    Traits::construct(alloc, buffer, counter);
    Traits::construct(alloc, buffer + 1, counter);
    assert(counter == 2);

    std::ranges::destroy_at(buffer);
    assert(counter == 1);
    std::ranges::destroy_at(buffer + 1);
    assert(counter == 0);

    Traits::deallocate(alloc, buffer, 2);
  }

  // Destroying a derived object with a virtual destructor.
  {
    std::allocator<VirtualCountedDerived> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    VirtualCountedDerived* buffer = Traits::allocate(alloc, 2);
    Traits::construct(alloc, buffer, counter);
    Traits::construct(alloc, buffer + 1, counter);
    assert(counter == 2);

    std::ranges::destroy_at(buffer);
    assert(counter == 1);
    std::ranges::destroy_at(buffer + 1);
    assert(counter == 0);

    Traits::deallocate(alloc, buffer, 2);
  }

  return true;
}

constexpr bool test_arrays() {
  // Pointer to an array.
  {
    using Array = Counted[3];
    std::allocator<Array> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Array* array = Traits::allocate(alloc, 1);
    Array& array_ref = *array;
    for (int i = 0; i != 3; ++i) {
      Traits::construct(alloc, std::addressof(array_ref[i]), counter);
    }
    assert(counter == 3);

    std::ranges::destroy_at(array);
    assert(counter == 0);

    Traits::deallocate(alloc, array, 1);
  }

  // Pointer to a two-dimensional array.
  {
    using Array = Counted[3][2];
    std::allocator<Array> alloc;
    using Traits = std::allocator_traits<decltype(alloc)>;
    int counter = 0;

    Array* array = Traits::allocate(alloc, 1);
    Array& array_ref = *array;
    for (int i = 0; i != 3; ++i) {
      for (int j = 0; j != 2; ++j) {
        Traits::construct(alloc, std::addressof(array_ref[i][j]), counter);
      }
    }
    assert(counter == 3 * 2);

    std::ranges::destroy_at(array);
    assert(counter == 0);

    Traits::deallocate(alloc, array, 1);
  }

  return true;
}

int main(int, char**) {
  test();
  test_arrays();

  static_assert(test());
  // TODO: Until std::construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());

  return 0;
}
