//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// type_traits

// is_nothrow_invocable

#include <type_traits>
#include <functional>
#include <vector>

#include "test_macros.h"

struct Tag {};

struct Implicit {
  Implicit(int) noexcept {}
};

struct ThrowsImplicit {
  ThrowsImplicit(int) {}
};

struct Explicit {
  explicit Explicit(int) noexcept {}
};

template <bool IsNoexcept, class Ret, class... Args>
struct CallObject {
  Ret operator()(Args&&...) const noexcept(IsNoexcept);
};

struct Sink {
  template <class... Args>
  void operator()(Args&&...) const noexcept {}
};

template <class Fn, class... Args>
constexpr bool throws_invocable() {
  return std::is_invocable<Fn, Args...>::value &&
         !std::is_nothrow_invocable<Fn, Args...>::value;
}

template <class Ret, class Fn, class... Args>
constexpr bool throws_invocable_r() {
  return std::is_invocable_r<Ret, Fn, Args...>::value &&
         !std::is_nothrow_invocable_r<Ret, Fn, Args...>::value;
}

void test_noexcept_function_pointers() {
  struct Dummy {
    void foo() noexcept {}
    static void bar() noexcept {}
  };
  // Check that PMF's and function pointers actually work and that
  // is_nothrow_invocable returns true for noexcept PMF's and function
  // pointers.
  static_assert(std::is_nothrow_invocable<decltype(&Dummy::foo), Dummy&>::value, "");
  static_assert(std::is_nothrow_invocable<decltype(&Dummy::bar)>::value, "");
}

int main(int, char**) {
  using AbominableFunc = void(...) const noexcept;
  //  Non-callable things
  {
    static_assert(!std::is_nothrow_invocable<void>::value, "");
    static_assert(!std::is_nothrow_invocable<const void>::value, "");
    static_assert(!std::is_nothrow_invocable<volatile void>::value, "");
    static_assert(!std::is_nothrow_invocable<const volatile void>::value, "");
    static_assert(!std::is_nothrow_invocable<std::nullptr_t>::value, "");
    static_assert(!std::is_nothrow_invocable<int>::value, "");
    static_assert(!std::is_nothrow_invocable<double>::value, "");

    static_assert(!std::is_nothrow_invocable<int[]>::value, "");
    static_assert(!std::is_nothrow_invocable<int[3]>::value, "");

    static_assert(!std::is_nothrow_invocable<int*>::value, "");
    static_assert(!std::is_nothrow_invocable<const int*>::value, "");
    static_assert(!std::is_nothrow_invocable<int const*>::value, "");

    static_assert(!std::is_nothrow_invocable<int&>::value, "");
    static_assert(!std::is_nothrow_invocable<const int&>::value, "");
    static_assert(!std::is_nothrow_invocable<int&&>::value, "");

    static_assert(!std::is_nothrow_invocable<int, std::vector<int> >::value,
                  "");
    static_assert(!std::is_nothrow_invocable<int, std::vector<int*> >::value,
                  "");
    static_assert(!std::is_nothrow_invocable<int, std::vector<int**> >::value,
                  "");

    static_assert(!std::is_nothrow_invocable<AbominableFunc>::value, "");

    //  with parameters
    static_assert(!std::is_nothrow_invocable<int, int>::value, "");
    static_assert(!std::is_nothrow_invocable<int, double, float>::value, "");
    static_assert(!std::is_nothrow_invocable<int, char, float, double>::value,
                  "");
    static_assert(!std::is_nothrow_invocable<Sink, AbominableFunc>::value, "");
    static_assert(!std::is_nothrow_invocable<Sink, void>::value, "");
    static_assert(!std::is_nothrow_invocable<Sink, const volatile void>::value,
                  "");

    static_assert(!std::is_nothrow_invocable_r<int, void>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, const void>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, volatile void>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, const volatile void>::value,
                  "");
    static_assert(!std::is_nothrow_invocable_r<int, std::nullptr_t>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, int>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, double>::value, "");

    static_assert(!std::is_nothrow_invocable_r<int, int[]>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, int[3]>::value, "");

    static_assert(!std::is_nothrow_invocable_r<int, int*>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, const int*>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, int const*>::value, "");

    static_assert(!std::is_nothrow_invocable_r<int, int&>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, const int&>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, int&&>::value, "");

    static_assert(!std::is_nothrow_invocable_r<int, std::vector<int> >::value,
                  "");
    static_assert(!std::is_nothrow_invocable_r<int, std::vector<int*> >::value,
                  "");
    static_assert(!std::is_nothrow_invocable_r<int, std::vector<int**> >::value,
                  "");
    static_assert(!std::is_nothrow_invocable_r<void, AbominableFunc>::value,
                  "");

    //  with parameters
    static_assert(!std::is_nothrow_invocable_r<int, int, int>::value, "");
    static_assert(!std::is_nothrow_invocable_r<int, int, double, float>::value,
                  "");
    static_assert(
        !std::is_nothrow_invocable_r<int, int, char, float, double>::value, "");
    static_assert(
        !std::is_nothrow_invocable_r<void, Sink, AbominableFunc>::value, "");
    static_assert(!std::is_nothrow_invocable_r<void, Sink, void>::value, "");
    static_assert(
        !std::is_nothrow_invocable_r<void, Sink, const volatile void>::value,
        "");
  }

  {
    // Check that the conversion to the return type is properly checked
    using Fn = CallObject<true, int>;
    static_assert(std::is_nothrow_invocable_r<Implicit, Fn>::value, "");
    static_assert(std::is_nothrow_invocable_r<double, Fn>::value, "");
    static_assert(std::is_nothrow_invocable_r<const volatile void, Fn>::value,
                  "");
    static_assert(throws_invocable_r<ThrowsImplicit, Fn>(), "");
    static_assert(!std::is_nothrow_invocable<Fn(), Explicit>(), "");
  }
  {
    // Check that the conversion to the parameters is properly checked
    using Fn = CallObject<true, void, const Implicit&, const ThrowsImplicit&>;
    static_assert(
        std::is_nothrow_invocable<Fn, Implicit&, ThrowsImplicit&>::value, "");
    static_assert(std::is_nothrow_invocable<Fn, int, ThrowsImplicit&>::value,
                  "");
    static_assert(throws_invocable<Fn, int, int>(), "");
    static_assert(!std::is_nothrow_invocable<Fn>::value, "");
  }
  {
    // Check that the noexcept-ness of function objects is checked.
    using Fn = CallObject<true, void>;
    using Fn2 = CallObject<false, void>;
    static_assert(std::is_nothrow_invocable<Fn>::value, "");
    static_assert(throws_invocable<Fn2>(), "");
  }
  {
    // Check that PMD derefs are noexcept
    using Fn = int(Tag::*);
    static_assert(std::is_nothrow_invocable<Fn, Tag&>::value, "");
    static_assert(std::is_nothrow_invocable_r<Implicit, Fn, Tag&>::value, "");
    static_assert(throws_invocable_r<ThrowsImplicit, Fn, Tag&>(), "");
  }
  {
    // Check that it's fine if the result type is non-moveable.
    struct CantMove {
      CantMove() = default;
      CantMove(CantMove&&) = delete;
    };

    static_assert(!std::is_move_constructible_v<CantMove>);
    static_assert(!std::is_copy_constructible_v<CantMove>);

    using Fn = CantMove() noexcept;

    static_assert(std::is_nothrow_invocable_r<CantMove, Fn>::value);
    static_assert(!std::is_nothrow_invocable_r<CantMove, Fn, int>::value);

    static_assert(std::is_nothrow_invocable_r_v<CantMove, Fn>);
    static_assert(!std::is_nothrow_invocable_r_v<CantMove, Fn, int>);
  }
  {
    // Check for is_nothrow_invocable_v
    using Fn = CallObject<true, int>;
    static_assert(std::is_nothrow_invocable_v<Fn>, "");
    static_assert(!std::is_nothrow_invocable_v<Fn, int>, "");
  }
  {
    // Check for is_nothrow_invocable_r_v
    using Fn = CallObject<true, int>;
    static_assert(std::is_nothrow_invocable_r_v<void, Fn>, "");
    static_assert(!std::is_nothrow_invocable_r_v<int, Fn, int>, "");
  }
  test_noexcept_function_pointers();

  return 0;
}
