//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// In case the function signature is taking an argument by value,
// when the type is small and trivial, we pass it internally by value,
// otherwise, we pass it by rvalue reference

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

struct Small {};

struct Big {
  char c[128];
};

struct SmallButNonTrivial {
  Small s;
  SmallButNonTrivial() = default;
  SmallButNonTrivial(const SmallButNonTrivial&) {}
};

// Need to inspect the internal state of function_ref, and our implementation friend-ed all the function_ref
// specializations, so we can just specialize an undefined function_ref here to inspect the defined ones
template <>
class std::function_ref<int> {
  using storage = std::__function_ref_storage;
  void test() {
    {
      // by value small argument
      // the internal function should pass by value
      using call = std::function_ref<void(Small)>::__call_t;
      static_assert(std::is_same_v<call, void (*)(storage, Small)>);
    }
    {
      // by value big argument
      // the internal function should pass by rvalue reference
      using call = std::function_ref<void(Big)>::__call_t;
      static_assert(std::is_same_v<call, void (*)(storage, Big&&)>);
    }
    {
      // by value non-trivial small argument
      // the internal function should pass by rvalue reference to avoid unnecessary copy/move
      using call = std::function_ref<void(SmallButNonTrivial)>::__call_t;
      static_assert(std::is_same_v<call, void (*)(storage, SmallButNonTrivial&&)>);
    }
    {
      // by lvalue reference argument
      // the internal function should pass by lvalue reference
      using call = std::function_ref<void(Small&)>::__call_t;
      static_assert(std::is_same_v<call, void (*)(storage, Small&)>);
    }
    {
      // by rvalue reference argument
      // the internal function should pass by rvalue reference
      using call = std::function_ref<void(Small&&)>::__call_t;
      static_assert(std::is_same_v<call, void (*)(storage, Small&&)>);
    }
  }
};

struct TrackCopyMove {
  mutable int copy_count = 0;
  int move_count         = 0;

  TrackCopyMove() = default;
  TrackCopyMove(const TrackCopyMove& other) : copy_count(other.copy_count), move_count(other.move_count) {
    ++copy_count;
    ++other.copy_count;
  }

  TrackCopyMove(TrackCopyMove&& other) noexcept : copy_count(other.copy_count), move_count(other.move_count) {
    ++move_count;
    ++other.move_count;
  }
  TrackCopyMove& operator=(const TrackCopyMove& other) {
    ++copy_count;
    ++other.copy_count;
    return *this;
  }
  TrackCopyMove& operator=(TrackCopyMove&& other) noexcept {
    ++move_count;
    ++other.move_count;
    return *this;
  }
};

void test() {
  {
    // Arg type is an lvalue reference, we should not copy or move the object
    TrackCopyMove t;
    auto lambda = [&t](TrackCopyMove& tm) {
      assert(&tm == &t);
      assert(tm.copy_count == 0);
      assert(tm.move_count == 0);
    };
    std::function_ref<void(TrackCopyMove&)> f = lambda;
    f(t);
  }
  {
    // Arg type is an rvalue reference, we should not copy or move the object
    TrackCopyMove t;
    auto lambda = [&t](TrackCopyMove&& tm) {
      assert(&tm == &t);
      assert(tm.copy_count == 0);
      assert(tm.move_count == 0);
    };
    std::function_ref<void(TrackCopyMove&&)> f = lambda;
    f(std::move(t));
  }
  {
    // Arg type is a prvalue, we should move but not copy the object
    // In this case, where the type is not trivially copyable, the object should be
    // moved exactly once when passing into the lambda. The internal functions
    // of function_ref should forward the argument without copying or moving it
    auto lambda = [](TrackCopyMove tm) {
      assert(tm.copy_count == 0);
      assert(tm.move_count == 1);
    };
    std::function_ref<void(TrackCopyMove)> f = lambda;
    f(TrackCopyMove{});
  }
}

int main(int, char**) {
  test();

  return 0;
}
