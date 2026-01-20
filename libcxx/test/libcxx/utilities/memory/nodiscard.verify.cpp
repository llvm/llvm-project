//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that functions are marked [[nodiscard]]

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_RAW_STORAGE_ITERATOR
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>

#include "test_macros.h"

void test() {
  int i = 0;

  {
    std::addressof(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::get_temporary_buffer<int>(0);
#if _LIBCPP_STD_VER >= 26
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::is_sufficiently_aligned<2>(&i);
#endif
  }

  {
    struct Alloc {
      using value_type = int;

      value_type* allocate(std::size_t) { return nullptr; }
    } allocator;
    using AllocTraits = std::allocator_traits<Alloc>;

    struct HintedAlloc {
      using value_type         = int;
      using size_type          = std::size_t;
      using const_void_pointer = const void*;

      value_type* allocate(size_type) { return nullptr; }
      value_type* allocate(size_type, const_void_pointer) { return nullptr; }
    } hintedAllocator;
    using HintedAllocTraits = std::allocator_traits<HintedAlloc>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    AllocTraits::allocate(allocator, 1);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    HintedAllocTraits::allocate(hintedAllocator, 1, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    AllocTraits::allocate(allocator, 1, nullptr);

#if TEST_STD_VER >= 23
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    AllocTraits::allocate_at_least(allocator, 1);
#endif

    struct SizedAlloc {
      using value_type = int;
      using size_type  = std::size_t;

      value_type* allocate(std::size_t) { return nullptr; }
      value_type* allocate(std::size_t, const void*) { return nullptr; }

      size_type max_size() const { return 0; }
    } sizedAllocator;
    using SizedAllocTraits = std::allocator_traits<SizedAlloc>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    SizedAllocTraits::max_size(sizedAllocator);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    AllocTraits::max_size(allocator);

    struct SelectAlloc {
      using value_type         = int;
      using const_void_pointer = const void*;

      value_type* allocate(std::size_t) { return nullptr; }
      value_type* allocate(std::size_t, const void*) { return nullptr; }

      SelectAlloc select_on_container_copy_construction() const { return SelectAlloc(); };
    } selectAllocator;
    using SelectAllocTraits = std::allocator_traits<SelectAlloc>;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    SelectAllocTraits::select_on_container_copy_construction(selectAllocator);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    AllocTraits::select_on_container_copy_construction(allocator);
  }

  {
    std::allocator<int> allocator;

    allocator.allocate(1); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 23
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator.allocate_at_least(1);
#endif

#if TEST_STD_VER <= 17
    const int ci = 0;

    allocator.address(i);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator.address(ci); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    allocator.allocate(1, nullptr);
    allocator.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  }

#if TEST_STD_VER >= 14
  {
    std::raw_storage_iterator<int*, int> it{nullptr};

    *it;       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
}
