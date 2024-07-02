//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr decltype(auto) operator*() const;

#include <ranges>

#include <array>
#include <cassert>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../types.h"

constexpr bool test() {
  { // Result of `operator*` is (maybe const) lvalue reference
    using V       = std::ranges::owning_view<std::vector<std::string>>;
    using Pattern = std::ranges::owning_view<std::string>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{"ab", "cd", "ef"}}, Pattern{"><"});

    {
      auto it                                  = jwv.begin();
      std::same_as<char&> decltype(auto) v_ref = *std::as_const(it);
      assert(v_ref == 'a');
      std::ranges::advance(it, 2);
      std::same_as<char&> decltype(auto) pattern_ref = *it;
      assert(pattern_ref == '>');
    }

    {
      auto cit                                        = std::as_const(jwv).begin();
      std::same_as<const char&> decltype(auto) cv_ref = *cit;
      assert(cv_ref == 'a');
      std::ranges::advance(cit, 3);
      std::same_as<const char&> decltype(auto) cpattern_ref = *std::as_const(cit);
      assert(cpattern_ref == '<');
    }
  }

  { // Result of `operator*` is const lvalue reference
    using V       = std::ranges::owning_view<std::vector<std::string_view>>;
    using Pattern = std::string_view;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    JWV jwv(V{{"123", "456", "789"}}, Pattern{"._."});

    {
      auto it                                        = jwv.begin();
      std::same_as<const char&> decltype(auto) v_ref = *it;
      assert(v_ref == '1');
      std::ranges::advance(it, 3);
      std::same_as<const char&> decltype(auto) pattern_ref = *std::as_const(it);
      assert(pattern_ref == '.');
    }

    {
      auto cit                                        = std::as_const(jwv).begin();
      std::same_as<const char&> decltype(auto) cv_ref = *std::as_const(cit);
      assert(cv_ref == '1');
      std::ranges::advance(cit, 4);
      std::same_as<const char&> decltype(auto) cpattern_ref = *cit;
      assert(cpattern_ref == '_');
    }
  }

  { // Result of `operator*` is prvalue
    using V       = std::vector<std::string_view>;
    using Pattern = RvalueVector<char>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    JWV jwv(V{"x^2", "y^2", "z^2"}, Pattern{{' ', '+', ' '}});

    {
      auto it                                 = jwv.begin();
      std::same_as<char> decltype(auto) v_ref = *std::as_const(it);
      assert(v_ref == 'x');
      std::ranges::advance(it, 3);
      std::same_as<char> decltype(auto) pattern_ref = *it;
      assert(pattern_ref == ' ');
    }

    {
      auto cit                                 = std::as_const(jwv).begin();
      std::same_as<char> decltype(auto) cv_ref = *cit;
      assert(cv_ref == 'x');
      std::ranges::advance(cit, 4);
      std::same_as<char> decltype(auto) cpattern_ref = *std::as_const(cit);
      assert(cpattern_ref == '+');
    }
  }

  { // Result of `operator*` is (maybe const) rvalue reference
    using Inner   = std::ranges::as_rvalue_view<std::ranges::owning_view<std::string>>;
    using V       = std::ranges::owning_view<std::vector<Inner>>;
    using Pattern = std::ranges::as_rvalue_view<std::ranges::owning_view<std::array<char, 2>>>;
    using JWV     = std::ranges::join_with_view<V, Pattern>;

    std::vector<Inner> vec;
    vec.emplace_back(Inner{{"x*y"}});
    vec.emplace_back(Inner{{"y*z"}});
    vec.emplace_back(Inner{{"z*x"}});
    JWV jwv(V(std::move(vec)), Pattern(std::array{',', ' '}));

    {
      auto it                                   = jwv.begin();
      std::same_as<char&&> decltype(auto) v_ref = *it;
      assert(v_ref == 'x');
      std::ranges::advance(it, 3);
      std::same_as<char&&> decltype(auto) pattern_ref = *std::as_const(it);
      assert(pattern_ref == ',');
    }

    {
      auto cit                                         = std::as_const(jwv).begin();
      std::same_as<const char&&> decltype(auto) cv_ref = *std::as_const(cit);
      assert(cv_ref == 'x');
      std::ranges::advance(cit, 4);
      std::same_as<const char&&> decltype(auto) cpattern_ref = *cit;
      assert(cpattern_ref == ' ');
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
