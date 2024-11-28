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
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../types.h"

struct ProxyInt {
  int& val;
};

class ExtraProxyInt {
public:
  constexpr ExtraProxyInt(ProxyInt i) : val(i.val) {}
  constexpr ExtraProxyInt(int i) : val(i) {}

  constexpr int get() const { return val; }

private:
  int val;
};

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<ProxyInt, int, TQual, UQual> {
  using type = ExtraProxyInt;
};

template <template <class> class TQual, template <class> class UQual>
struct std::basic_common_reference<int, ProxyInt, TQual, UQual> {
  using type = ExtraProxyInt;
};

static_assert(std::common_reference_with<int&, ProxyInt>);
static_assert(std::common_reference_with<int&, ExtraProxyInt>);

class ProxyIter {
public:
  using value_type      = int;
  using difference_type = std::ptrdiff_t;

  constexpr ProxyIter() : ptr_(nullptr) {}
  constexpr explicit ProxyIter(int* p) : ptr_(p) {}

  constexpr ProxyInt operator*() const { return ProxyInt{*ptr_}; }

  constexpr ProxyIter& operator++() {
    ++ptr_;
    return *this;
  }

  constexpr ProxyIter operator++(int) {
    ProxyIter tmp = *this;
    ++ptr_;
    return tmp;
  }

  constexpr bool operator==(const ProxyIter& other) const { return ptr_ == other.ptr_; }

private:
  int* ptr_;
};

static_assert(std::forward_iterator<ProxyIter>);

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

  { // Result of `operator*` is type different from range_reference_t<InnerRng> and range_reference_t<Pattern>
    using Inner   = std::vector<int>;
    using V       = std::vector<Inner>;
    using Pattern = std::ranges::subrange<ProxyIter, ProxyIter>;
    using JWV     = std::ranges::join_with_view<std::ranges::owning_view<V>, std::ranges::owning_view<Pattern>>;

    static_assert(!std::same_as<std::ranges::range_reference_t<V>, std::ranges::range_reference_t<JWV>>);
    static_assert(!std::same_as<std::ranges::range_reference_t<Pattern>, std::ranges::range_reference_t<JWV>>);

    std::array<int, 2> pattern = {-1, -1};
    Pattern pattern_as_subrange(ProxyIter{pattern.data()}, ProxyIter{pattern.data() + pattern.size()});

    JWV jwv(V{Inner{1, 1}, Inner{2, 2}, Inner{3, 3}}, std::move(pattern_as_subrange));

    auto it                                          = jwv.begin();
    std::same_as<ExtraProxyInt> decltype(auto) v_ref = *it;
    assert(v_ref.get() == 1);
    std::ranges::advance(it, 7);
    std::same_as<ExtraProxyInt> decltype(auto) pattern_ref = *std::as_const(it);
    assert(pattern_ref.get() == -1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
