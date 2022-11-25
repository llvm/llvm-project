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

      constexpr const int& lvalue_proj() const { return i; }
      constexpr int prvalue_proj() const { return i; }
    };

    struct Comp {
      constexpr bool operator()(const int& lhs, const int& rhs) const { return lhs < rhs; }
      constexpr bool operator()(int&& lhs, int&& rhs) const { return lhs > rhs; }
    };

    auto val = S{10};
    auto low = S{20};
    auto high = S{30};
    // Check that the value category of the projection return type is preserved.
    assert(&std::ranges::clamp(val, low, high, Comp{}, &S::lvalue_proj) == &low);
    assert(&std::ranges::clamp(val, high, low, Comp{}, &S::prvalue_proj) == &low);
  }

  { // Check that the implementation doesn't cause double moves (which could result from calling the projection on
    // `value` once and then forwarding the result into the comparator).
    struct CheckDoubleMove {
      int i;
      bool moved = false;

      constexpr explicit CheckDoubleMove(int set_i) : i(set_i) {}
      constexpr CheckDoubleMove(const CheckDoubleMove&) = default;
      constexpr CheckDoubleMove(CheckDoubleMove&& rhs) noexcept : i(rhs.i) {
        assert(!rhs.moved);
        rhs.moved = true;
      }
    };

    auto val = CheckDoubleMove{20};
    auto low = CheckDoubleMove{10};
    auto high = CheckDoubleMove{30};

    auto moving_comp = [](CheckDoubleMove lhs, CheckDoubleMove rhs) { return lhs.i < rhs.i; };
    auto prvalue_proj = [](const CheckDoubleMove& x) -> CheckDoubleMove { return x; };
    assert(&std::ranges::clamp(val, low, high, moving_comp, prvalue_proj) == &val);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
