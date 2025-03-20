//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <memory>

// To allow checking that self-move works correctly.
// ADDITIONAL_COMPILE_FLAGS: -Wno-self-move

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// template<class _Alloc>
// struct __allocation_guard;

#include <__memory/allocation_guard.h>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_allocator.h"

using A = test_allocator<int>;

// A trimmed-down version of `test_allocator` that is copy-assignable (in general allocators don't have to support copy
// assignment).
template <class T>
struct AssignableAllocator {
  using size_type = unsigned;
  using difference_type = int;
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = typename std::add_lvalue_reference<value_type>::type;
  using const_reference = typename std::add_lvalue_reference<const value_type>::type;

  template <class U>
  struct rebind {
    using other = test_allocator<U>;
  };

  test_allocator_statistics* stats_ = nullptr;

  explicit AssignableAllocator(test_allocator_statistics& stats) : stats_(&stats) {
    ++stats_->count;
  }

  TEST_CONSTEXPR_CXX14 AssignableAllocator(const AssignableAllocator& rhs) TEST_NOEXCEPT
      : stats_(rhs.stats_) {
    if (stats_ != nullptr) {
      ++stats_->count;
      ++stats_->copied;
    }
  }

  TEST_CONSTEXPR_CXX14 AssignableAllocator& operator=(const AssignableAllocator& rhs) TEST_NOEXCEPT {
    stats_ = rhs.stats_;
    if (stats_ != nullptr) {
      ++stats_->count;
      ++stats_->copied;
    }

    return *this;
  }

  TEST_CONSTEXPR_CXX14 pointer allocate(size_type n, const void* = nullptr) {
    if (stats_ != nullptr) {
      ++stats_->alloc_count;
    }
    return std::allocator<value_type>().allocate(n);
  }

  TEST_CONSTEXPR_CXX14 void deallocate(pointer p, size_type s) {
    if (stats_ != nullptr) {
      --stats_->alloc_count;
    }
    std::allocator<value_type>().deallocate(p, s);
  }

  TEST_CONSTEXPR size_type max_size() const TEST_NOEXCEPT { return UINT_MAX / sizeof(T); }

  template <class U>
  TEST_CONSTEXPR_CXX20 void construct(pointer p, U&& val) {
    if (stats_ != nullptr)
      ++stats_->construct_count;
#if TEST_STD_VER > 17
    std::construct_at(std::to_address(p), std::forward<U>(val));
#else
    ::new (static_cast<void*>(p)) T(std::forward<U>(val));
#endif
  }

  TEST_CONSTEXPR_CXX14 void destroy(pointer p) {
    if (stats_ != nullptr) {
      ++stats_->destroy_count;
    }
    p->~T();
  }
};

// Move-only.
static_assert(!std::is_copy_constructible<std::__allocation_guard<A> >::value, "");
static_assert(std::is_move_constructible<std::__allocation_guard<A> >::value, "");
static_assert(!std::is_copy_assignable<std::__allocation_guard<A> >::value, "");
static_assert(std::is_move_assignable<std::__allocation_guard<A> >::value, "");

int main(int, char**) {
  const int size = 42;

  { // The constructor allocates using the given allocator.
    test_allocator_statistics stats;
    std::__allocation_guard<A> guard(A(&stats), size);
    assert(stats.alloc_count == 1);
    assert(guard.__get() != nullptr);
  }

  { // The destructor deallocates using the given allocator.
    test_allocator_statistics stats;
    {
      std::__allocation_guard<A> guard(A(&stats), size);
      assert(stats.alloc_count == 1);
    }
    assert(stats.alloc_count == 0);
  }

  { // `__release_ptr` prevents deallocation.
    test_allocator_statistics stats;
    A alloc(&stats);
    int* ptr = nullptr;
    {
      std::__allocation_guard<A> guard(alloc, size);
      assert(stats.alloc_count == 1);
      ptr = guard.__release_ptr();
    }
    assert(stats.alloc_count == 1);
    alloc.deallocate(ptr, size);
  }

  { // Using the move constructor doesn't lead to double deletion.
    test_allocator_statistics stats;
    {
      std::__allocation_guard<A> guard1(A(&stats), size);
      assert(stats.alloc_count == 1);
      auto* ptr1 = guard1.__get();

      std::__allocation_guard<A> guard2 = std::move(guard1);
      assert(stats.alloc_count == 1);
      assert(guard1.__get() == nullptr);
      assert(guard2.__get() == ptr1);
    }
    assert(stats.alloc_count == 0);
  }

  { // Using the move assignment operator doesn't lead to double deletion.
    using A2 = AssignableAllocator<int>;

    test_allocator_statistics stats;
    {
      std::__allocation_guard<A2> guard1(A2(stats), size);
      assert(stats.alloc_count == 1);
      std::__allocation_guard<A2> guard2(A2(stats), size);
      assert(stats.alloc_count == 2);
      auto* ptr1 = guard1.__get();

      guard2 = std::move(guard1);
      assert(stats.alloc_count == 1);
      assert(guard1.__get() == nullptr);
      assert(guard2.__get() == ptr1);
    }
    assert(stats.alloc_count == 0);
  }

  { // Self-assignment is a no-op.
    using A2 = AssignableAllocator<int>;

    test_allocator_statistics stats;
    {
      std::__allocation_guard<A2> guard(A2(stats), size);
      assert(stats.alloc_count == 1);
      auto* ptr = guard.__get();

      guard = std::move(guard);
      assert(stats.alloc_count == 1);
      assert(guard.__get() == ptr);
    }
    assert(stats.alloc_count == 0);
  }

  return 0;
}
