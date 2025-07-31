//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
Allocator class useful for testing various propagation, always-equal scenarios.
*/

#ifndef _LIBCPP_STACKTRACE_TEST_ALLOCS_H
#define _LIBCPP_STACKTRACE_TEST_ALLOCS_H

#include <memory>
#include <type_traits>

template <typename T, bool _KNoExCtors, bool _KNoExAlloc, bool _KPropagate, bool _KAlwaysEqual>
struct TestAlloc {
  using size_type     = size_t;
  using value_type    = T;
  using pointer       = T*;
  using const_pointer = T const*;

  using Self = TestAlloc<T, _KNoExCtors, _KNoExAlloc, _KPropagate, _KAlwaysEqual>;

  template <typename U>
  using Other = TestAlloc<U, _KNoExCtors, _KNoExAlloc, _KPropagate, _KAlwaysEqual>;

  using propagate_on_container_copy_assignment = typename std::bool_constant<_KPropagate>;
  using propagate_on_container_move_assignment = typename std::bool_constant<_KPropagate>;
  using propagate_on_container_swap            = typename std::bool_constant<_KPropagate>;
  using is_always_equal                        = typename std::bool_constant<_KAlwaysEqual>;

  auto select_on_container_copy_construction(this auto& self) { return _KPropagate ? self : Self(); }

  template <typename U>
  struct rebind {
    using other = Other<U>;
  };

  static std::shared_ptr<std::allocator<std::byte>> new_arena() {
    return std::make_shared<std::allocator<std::byte>>();
  }

  static std::shared_ptr<std::allocator<std::byte>> global_arena() {
    static auto ret = new_arena();
    return ret;
  }

  /** Type-erased allocator used for servicing allocate and deallocate.
  Two `TestAlloc`'s are equal IFF they contain the same arena pointer.
  Always-equal `TestAlloc`'s get a pointer to a shared `global_arena`. */
  std::shared_ptr<std::allocator<std::byte>> arena_;

  /** Instances are equal IFF they have the same arena pointer (even if this is "always_equals",
  since such instances point to the global arena). */
  bool operator==(auto const& rhs) const noexcept { return arena_.get() == rhs.arena_.get(); }

  /** Construct with a new arena, or, if always-equal, the global arena. */
  TestAlloc() noexcept(_KNoExCtors) : arena_(_KAlwaysEqual ? global_arena() : new_arena()) {}

  template <typename U>
  TestAlloc(Other<U> const& rhs) : arena_(rhs.arena_) {}

  template <typename U>
  TestAlloc& operator=(Other<U> const& rhs) {
    arena_ = rhs.arena_;
  }

  std::allocator<T>& arena() { return *(std::allocator<T>*)arena_.get(); }

  T* allocate(size_t n) noexcept(_KNoExAlloc) { return arena().allocate(n); }
  auto allocate_at_least(size_t n) noexcept(_KNoExAlloc) { return arena().allocate_at_least(n); }
  void deallocate(T* ptr, size_t n) noexcept(_KNoExAlloc) { return arena().deallocate(ptr, n); }
};

// For convenience and readability:

template <typename T>
using AllocPropagate =
    TestAlloc<T,
              /*_KNoExCtors=*/true,
              /*_KNoExAlloc=*/true,
              /*_KPropagate=*/true,
              /*_KAlwaysEqual=*/false>;

template <typename T>
using AllocNoPropagate =
    TestAlloc<T,
              /*_KNoExCtors=*/true,
              /*_KNoExAlloc=*/true,
              /*_KPropagate=*/false,
              /*_KAlwaysEqual=*/false>;

template <typename T>
using AllocAlwaysEqual =
    TestAlloc<T,
              /*_KNoExCtors=*/true,
              /*_KNoExAlloc=*/true,
              /*_KPropagate=*/true,
              /*_KAlwaysEqual=*/true>;

#endif
