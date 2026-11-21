//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// promise_type::operator new(size_t, ...);
// promise_type::operator delete(void*, size_t);

#include <generator>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <memory_resource>
#include <new>
#include <type_traits>
#include <utility>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class P>
concept HasDefaultAllocation = requires { P::operator new(std::size_t(1)); };

static_assert(HasDefaultAllocation<std::generator<int>::promise_type>);
static_assert(HasDefaultAllocation<std::pmr::generator<int>::promise_type>);
static_assert(!HasDefaultAllocation<std::generator<int, void, no_default_allocator<int>>::promise_type>);

// An allocator with observable identity and alignment greater than the allocation block's alignment.
struct AllocationState {
  void* pointer     = nullptr;
  std::size_t count = 0;
};

template <class T>
struct alignas(64) AlignedAllocator {
  using value_type = T;
  AllocationState* state;
  explicit AlignedAllocator(AllocationState& s) : state(&s) {}
  template <class U>
  AlignedAllocator(const AlignedAllocator<U>& other) noexcept : state(other.state) {}
  T* allocate(std::size_t n) {
    assert(state->pointer == nullptr);
    auto p         = std::allocator<T>{}.allocate(n);
    state->pointer = p;
    state->count   = n;
    return p;
  }
  void deallocate(T* p, std::size_t n) {
    assert(state->pointer == p && state->count == n);
    state->pointer = nullptr;
    std::allocator<T>{}.deallocate(p, n);
  }
  template <class U>
  bool operator==(const AlignedAllocator<U>& other) const {
    return state == other.state;
  }
};

template <class P, class Alloc>
void test_allocation(const Alloc& alloc) {
  struct Owner {
  } owner;
  // Direct calls prevent coroutine allocation elision from hiding these checks.
  // Include sizes that leave the deallocation function at an unaligned address.
  for (std::size_t size = 1; size <= 2 * __STDCPP_DEFAULT_NEW_ALIGNMENT__ + 1; ++size) {
    if constexpr (HasDefaultAllocation<P>) {
      void* p = P::operator new(size);
      assert(reinterpret_cast<std::uintptr_t>(p) % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0);
      std::memset(p, 0x42, size);
      P::operator delete(p, size);
    }
    void* p = P::operator new(size, std::allocator_arg, alloc, 1, 2);
    assert(reinterpret_cast<std::uintptr_t>(p) % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0);
    std::memset(p, 0x42, size);
    P::operator delete(p, size);
    p = P::operator new(size, owner, std::allocator_arg, alloc, 1, 2);
    assert(reinterpret_cast<std::uintptr_t>(p) % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0);
    std::memset(p, 0x42, size);
    P::operator delete(p, size);
  }
}

template <class Alloc>
void test_allocators(const Alloc& alloc) {
  test_allocation<typename std::generator<int, void, Alloc>::promise_type>(alloc);
  test_allocation<std::generator<int>::promise_type>(alloc);
}

template <class P>
void test_allocator_state(const test_allocator<int>& alloc, test_allocator_statistics& stats) {
  struct Owner {
  } owner;
  for (bool member : {false, true}) {
    void* p =
        member ? P::operator new(17, owner, std::allocator_arg, alloc) : P::operator new(17, std::allocator_arg, alloc);
    assert(stats.alloc_count == 1 && stats.allocated_size > 0);
    P::operator delete(p, 17);
    assert(stats.alloc_count == 0 && stats.allocated_size == 0 && stats.count == 1);
  }
}

template <class G, class Alloc>
G generate(std::allocator_arg_t, Alloc, int value) {
  co_yield value;
}

struct Owner {
  template <class G, class Alloc>
  G generate(std::allocator_arg_t, Alloc, int value) {
    co_yield value;
  }
};

class Resource : public std::pmr::memory_resource {
public:
  int outstanding = 0;
  int allocations = 0;

private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    ++outstanding;
    ++allocations;
    return p;
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    --outstanding;
    assert(outstanding >= 0);
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
};

template <class G, class Alloc>
void test_coroutine(const Alloc& alloc) {
  {
    auto g = generate<G>(std::allocator_arg, alloc, 42);
    auto i = g.begin();
    assert(*i == 42);
    ++i;
    assert(i == g.end());
  }
  {
    Owner owner;
    auto g = owner.generate<G>(std::allocator_arg, alloc, 43);
    auto i = g.begin();
    assert(*i == 43);
    // Destroy while suspended.
  }
}

int main(int, char**) {
  test_allocators(std::allocator<std::byte>{});
  test_allocators(std::allocator<float>{});
  test_allocators(bare_allocator<int>{});
  test_allocators(no_default_allocator<int>::create());

  test_allocator_statistics stats;
  {
    ::test_allocator<int> alloc(42, &stats);
    test_allocators(alloc);
    assert(stats.alloc_count == 0 && stats.allocated_size == 0 && stats.count == 1);
    test_allocator_state<std::generator<int>::promise_type>(alloc, stats);
    test_allocator_state<std::generator<int, void, ::test_allocator<int>>::promise_type>(alloc, stats);
    test_coroutine<std::generator<int>>(alloc);
    test_coroutine<std::generator<int, void, ::test_allocator<int>>>(alloc);
    assert(stats.alloc_count == 0 && stats.allocated_size == 0 && stats.count == 1);

#ifndef TEST_HAS_NO_EXCEPTIONS
    stats.throw_after = stats.time_to_throw;
    using P           = std::generator<int>::promise_type;
    try {
      (void)P::operator new(17, std::allocator_arg, alloc);
      assert(false);
    } catch (const std::bad_alloc&) {
    }
    assert(stats.alloc_count == 0 && stats.allocated_size == 0 && stats.count == 1);
#endif
  }
  assert(stats.count == 0);

  AllocationState state;
  AlignedAllocator<int> aligned(state);
  test_allocators(aligned);
  assert(state.pointer == nullptr);

  Resource resource;
  std::pmr::polymorphic_allocator<> alloc(&resource);
  test_allocators(alloc);
  assert(resource.allocations != 0 && resource.outstanding == 0);
  // Check conversion from memory_resource* without permitting allocation elision to hide a fallback.
  using P = std::pmr::generator<int>::promise_type;
  void* p = P::operator new(17, std::allocator_arg, &resource);
  assert(resource.outstanding == 1);
  P::operator delete(p, 17);
  assert(resource.outstanding == 0);
  test_coroutine<std::pmr::generator<int>>(alloc);
  test_coroutine<std::pmr::generator<int>>(&resource);
  test_coroutine<std::generator<int>>(alloc);
  assert(resource.outstanding == 0);
  return 0;
}
