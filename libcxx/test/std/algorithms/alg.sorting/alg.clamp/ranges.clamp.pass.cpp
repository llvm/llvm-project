//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class Proj = identity,
//          indirect_strict_weak_order<projected<const T*, Proj>> Comp = ranges::less>
//   constexpr const T&
//     ranges::clamp(const T& v, const T& lo, const T& hi, Comp comp = {}, Proj proj = {});

#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <iterator>
#include <utility>

template <class T, class Comp = std::ranges::less, class Proj = std::identity>
concept HasClamp =
    requires(T&& val, T&& low, T&& high, Comp&& comp, Proj&& proj) {
      std::ranges::clamp(std::forward<T>(val), std::forward<T>(low), std::forward<T>(high),
          std::forward<Comp>(comp), std::forward<Proj>(proj));
    };

struct NoComp {};
struct CreateNoComp {
  auto operator()(int) const { return NoComp(); }
};

static_assert(HasClamp<int, std::ranges::less, std::identity>);
static_assert(!HasClamp<NoComp>);
static_assert(!HasClamp<int, NoComp>);
static_assert(!HasClamp<int, std::ranges::less, CreateNoComp>);

struct EnsureValueCategoryComp {
  constexpr bool operator()(const int&& x, const int&& y) const { return x < y; }
  constexpr bool operator()(const int&& x, int& y) const { return x < y; }
  constexpr bool operator()(int& x, const int&& y) const { return x < y; }
  constexpr bool operator()(int& x, int& y) const { return x < y; }
  constexpr bool operator()(std::same_as<const int&> auto&& x, std::same_as<const int&> auto&& y) const {
    return x < y;
  }
};

constexpr bool test() {
  { // low < val < high
    int val = 2;
    int low = 1;
    int high = 3;
    std::same_as<const int&> decltype(auto) ret = std::ranges::clamp(val, low, high);
    assert(ret == 2);
    assert(&ret == &val);
  }

  { // low > val < high
    assert(std::ranges::clamp(10, 20, 30) == 20);
  }

  { // low < val > high
    assert(std::ranges::clamp(15, 5, 10) == 10);
  }

  { // low == val == high
    int val = 10;
    assert(&std::ranges::clamp(val, 10, 10) == &val);
  }

  { // Check that a custom comparator works.
    assert(std::ranges::clamp(10, 30, 20, std::ranges::greater{}) == 20);
  }

  { // Check that a custom projection works.
    struct S {
      int i;
      constexpr bool operator==(S const& other) const { return i == other.i; }
    };

    auto val = S{10};
    auto low = S{20};
    auto high = S{30};
    auto proj = [](S const& s) -> int const& { return s.i; };

    assert(std::ranges::clamp(val, low, high, std::less{}, proj) == low);
  }

  { // Ensure that we respect the value category of the projection when calling the comparator.
    // This additional example was provided by Tim Song in https://github.com/microsoft/STL/issues/3970#issuecomment-1685120958.
    struct MoveProj {
      constexpr int const&& operator()(int const& x) const { return std::move(x); }
    };

    static_assert(std::indirect_strict_weak_order<EnsureValueCategoryComp, std::projected<const int*, MoveProj>>);

    assert(std::ranges::clamp(0, 1, 2, EnsureValueCategoryComp{}, MoveProj{}) == 1);
  }

  { // Make sure we don't call the projection more than three times per [alg.clamp], see #64717
    int counter              = 0;
    auto projection_function = [&counter](const int value) -> int {
      counter++;
      return value;
    };
    assert(std::ranges::clamp(3, 2, 4, std::ranges::less{}, projection_function) == 3);
#if _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_EXTENSIVE && _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_DEBUG
    assert(counter <= 3);
#endif
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
