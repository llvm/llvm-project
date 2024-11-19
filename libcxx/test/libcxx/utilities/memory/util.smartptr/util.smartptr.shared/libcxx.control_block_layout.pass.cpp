//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-abi-no-compressed-pair-padding

// This test makes sure that the control block implementation used for non-array
// types in std::make_shared and std::allocate_shared is ABI compatible with the
// original implementation.
//
// This test is relevant because the implementation of that control block is
// different starting in C++20, a change that was required to implement P0674.

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <string>
#include <vector>

#include "test_macros.h"

struct value_init_tag {};

template <class T, int _Idx, bool CanBeEmptyBase = std::is_empty<T>::value && !std::__libcpp_is_final<T>::value>
struct compressed_pair_elem {
  explicit compressed_pair_elem(value_init_tag) : value_() {}

  template <class U>
  explicit compressed_pair_elem(U&& u) : value_(std::forward<U>(u)) {}

  T& get() { return value_; }

private:
  T value_;
};

template <class T, int _Idx>
struct compressed_pair_elem<T, _Idx, true> : private T {
  explicit compressed_pair_elem(value_init_tag) : T() {}

  template <class U>
  explicit compressed_pair_elem(U&& u) : T(std::forward<U>(u)) {}

  T& get() { return *this; }
};

template <class T1, class T2>
class compressed_pair : private compressed_pair_elem<T1, 0>, private compressed_pair_elem<T2, 1> {
public:
  using Base1 = compressed_pair_elem<T1, 0>;
  using Base2 = compressed_pair_elem<T2, 1>;

  template <class U1, class U2>
  explicit compressed_pair(U1&& t1, U2&& t2) : Base1(std::forward<U1>(t1)), Base2(std::forward<U2>(t2)) {}

  T1& first() { return static_cast<Base1&>(*this).get(); }
  T2& second() { return static_cast<Base2&>(*this).get(); }
};

// This is the pre-C++20 implementation of the control block used by non-array
// std::allocate_shared and std::make_shared. We keep it here so that we can
// make sure our implementation is backwards compatible with it forever.
//
// Of course, the class and its methods were renamed, but the size and layout
// of the class should remain the same as the original implementation.
template <class T, class Alloc>
struct OldEmplaceControlBlock : std::__shared_weak_count {
  explicit OldEmplaceControlBlock(Alloc a) : data_(std::move(a), value_init_tag()) {}
  T* get_elem() noexcept { return std::addressof(data_.second()); }
  Alloc* get_alloc() noexcept { return std::addressof(data_.first()); }

private:
  virtual void __on_zero_shared() noexcept {
    // Not implemented
  }

  virtual void __on_zero_shared_weak() noexcept {
    // Not implemented
  }

  compressed_pair<Alloc, T> data_;
};

template <class T, template <class> class Alloc>
void test() {
  using Old = OldEmplaceControlBlock<T, Alloc<T>>;
  using New = std::__shared_ptr_emplace<T, Alloc<T>>;

  static_assert(sizeof(New) == sizeof(Old), "");
  static_assert(alignof(New) == alignof(Old), "");

  // Also make sure each member is at the same offset
  Alloc<T> a;
  Old old(a);
  New new_(a);

  // 1. Check the stored object
  {
    char const* old_elem      = reinterpret_cast<char const*>(old.get_elem());
    char const* new_elem      = reinterpret_cast<char const*>(new_.__get_elem());
    std::ptrdiff_t old_offset = old_elem - reinterpret_cast<char const*>(&old);
    std::ptrdiff_t new_offset = new_elem - reinterpret_cast<char const*>(&new_);
    assert(new_offset == old_offset && "offset of stored element changed");
  }

  // 2. Check the allocator
  {
    char const* old_alloc     = reinterpret_cast<char const*>(old.get_alloc());
    char const* new_alloc     = reinterpret_cast<char const*>(new_.__get_alloc());
    std::ptrdiff_t old_offset = old_alloc - reinterpret_cast<char const*>(&old);
    std::ptrdiff_t new_offset = new_alloc - reinterpret_cast<char const*>(&new_);
    assert(new_offset == old_offset && "offset of allocator changed");
  }

  // Make sure both types have the same triviality (that has ABI impact since
  // it determined how objects are passed). Both should be non-trivial.
  static_assert(std::is_trivial<New>::value == std::is_trivial<Old>::value, "");
}

// Object types to store in the control block
struct TrivialEmptyType {};

struct alignas(32) OveralignedEmptyType {};

struct TrivialNonEmptyType {
  char c[11];
};

struct FinalEmptyType final {};

struct NonTrivialType {
  char c[22];
  NonTrivialType() : c{'x'} {}
};

struct VirtualFunctionType {
  virtual ~VirtualFunctionType() {}
};

// Allocator types
template <class T>
struct TrivialEmptyAlloc {
  using value_type    = T;
  TrivialEmptyAlloc() = default;
  template <class U>
  TrivialEmptyAlloc(TrivialEmptyAlloc<U>) {}
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) {}
};

template <class T>
struct TrivialNonEmptyAlloc {
  char storage[77];
  using value_type       = T;
  TrivialNonEmptyAlloc() = default;
  template <class U>
  TrivialNonEmptyAlloc(TrivialNonEmptyAlloc<U>) {}
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) {}
};

template <class T>
struct FinalEmptyAlloc final {
  using value_type  = T;
  FinalEmptyAlloc() = default;
  template <class U>
  FinalEmptyAlloc(FinalEmptyAlloc<U>) {}
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) {}
};

template <class T>
struct NonTrivialAlloc {
  char storage[88];
  using value_type = T;
  NonTrivialAlloc() {}
  template <class U>
  NonTrivialAlloc(NonTrivialAlloc<U>) {}
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) {}
};

int main(int, char**) {
  test<TrivialEmptyType, TrivialEmptyAlloc>();
  test<TrivialEmptyType, TrivialNonEmptyAlloc>();
  test<TrivialEmptyType, FinalEmptyAlloc>();
  test<TrivialEmptyType, NonTrivialAlloc>();

#if !defined(TEST_HAS_NO_ALIGNED_ALLOCATION)
  test<OveralignedEmptyType, TrivialEmptyAlloc>();
  test<OveralignedEmptyType, TrivialNonEmptyAlloc>();
  test<OveralignedEmptyType, FinalEmptyAlloc>();
  test<OveralignedEmptyType, NonTrivialAlloc>();
#endif

  test<TrivialNonEmptyType, TrivialEmptyAlloc>();
  test<TrivialNonEmptyType, TrivialNonEmptyAlloc>();
  test<TrivialNonEmptyType, FinalEmptyAlloc>();
  test<TrivialNonEmptyType, NonTrivialAlloc>();

  test<FinalEmptyType, TrivialEmptyAlloc>();
  // FinalEmptyType combined with TrivialNonEmptyAlloc, FinalEmptyAlloc or NonTrivialAlloc is known to have an ABI break
  // between LLVM 19 and LLVM 20. It's been deemed not severe enough to cause actual breakage.

  test<NonTrivialType, TrivialEmptyAlloc>();
  test<NonTrivialType, TrivialNonEmptyAlloc>();
  test<NonTrivialType, FinalEmptyAlloc>();
  test<NonTrivialType, NonTrivialAlloc>();

  test<VirtualFunctionType, TrivialEmptyAlloc>();
  test<VirtualFunctionType, TrivialNonEmptyAlloc>();
  test<VirtualFunctionType, FinalEmptyAlloc>();
  test<VirtualFunctionType, NonTrivialAlloc>();

  // Test a few real world types just to make sure we didn't mess up badly somehow
  test<std::string, std::allocator>();
  test<int, std::allocator>();
  test<std::vector<int>, std::allocator>();

  return 0;
}
