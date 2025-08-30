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

#include <cstddef>
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

  static std::shared_ptr<std::allocator<std::byte>> new_alloc() {
    return std::make_shared<std::allocator<std::byte>>();
  }

  static std::shared_ptr<std::allocator<std::byte>> global_alloc() {
    static auto ret = new_alloc();
    return ret;
  }

  /** Type-erased allocator used for servicing allocate and deallocate.
  Two `TestAlloc`'s are equal IFF they contain the same alloc pointer.
  Always-equal `TestAlloc`'s get a pointer to a shared `global_alloc`. */
  std::shared_ptr<std::allocator<std::byte>> alloc_;

  /** Instances are equal IFF they have the same alloc pointer (even if this is "always_equals",
  since such instances point to the global alloc). */
  bool operator==(auto const& rhs) const noexcept { return alloc_.get() == rhs.alloc_.get(); }

  /** Construct with a new alloc, or, if always-equal, the global alloc. */
  TestAlloc() noexcept(_KNoExCtors) : alloc_(_KAlwaysEqual ? global_alloc() : new_alloc()) {}

  template <typename U>
  TestAlloc(Other<U> const& rhs) : alloc_(rhs.alloc_) {}

  template <typename U>
  TestAlloc& operator=(Other<U> const& rhs) {
    alloc_ = rhs.alloc_;
  }

  std::allocator<T>& alloc() { return *(std::allocator<T>*)alloc_.get(); }

  T* allocate(size_t n) noexcept(_KNoExAlloc) { return alloc().allocate(n); }
  auto allocate_at_least(size_t n) noexcept(_KNoExAlloc) { return alloc().allocate_at_least(n); }
  void deallocate(T* ptr, size_t n) noexcept(_KNoExAlloc) { return alloc().deallocate(ptr, n); }
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
