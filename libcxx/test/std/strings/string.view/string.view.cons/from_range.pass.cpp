//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <string_view>

//  template <class Range>
//  constexpr basic_string_view(Range&& range);

#include <string_view>
#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "test_iterators.h"
#include "test_range.h"

template<class CharT>
constexpr void test() {
  auto data = MAKE_STRING_VIEW(CharT, "test");
  std::array<CharT, 4> arr;
  for(int i = 0; i < 4; ++i) {
    arr[i] = data[i];
  }
  auto sv = std::basic_string_view<CharT>(arr);

  ASSERT_SAME_TYPE(decltype(sv), std::basic_string_view<CharT>);
  assert(sv.size() == arr.size());
  assert(sv.data() == arr.data());
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();

  {
    struct NonConstConversionOperator {
      const char* data_ = "test";
      constexpr const char* begin() const { return data_; }
      constexpr const char* end() const { return data_ + 4; }
      constexpr operator std::basic_string_view<char>() { return "NonConstConversionOp"; }
    };

    NonConstConversionOperator nc;
    std::string_view sv = nc;
    assert(sv == "NonConstConversionOp");
    static_assert(!std::is_constructible_v<std::string_view,
                                           const NonConstConversionOperator&>); // conversion operator is non-const
  }

  {
    struct ConstConversionOperator {
      const char* data_ = "test";
      constexpr const char* begin() const { return data_; }
      constexpr const char* end() const { return data_ + 4; }
      constexpr operator std::basic_string_view<char>() const { return "ConstConversionOp"; }
    };
    ConstConversionOperator cv;
    std::basic_string_view<char> sv = cv;
    assert(sv == "ConstConversionOp");
  }

  struct DeletedConversionOperator {
    const char* data_ = "test";
    constexpr const char* begin() const { return data_; }
    constexpr const char* end() const { return data_ + 4; }
    operator std::basic_string_view<char>() = delete;
  };

  struct DeletedConstConversionOperator {
    const char* data_ = "test";
    constexpr const char* begin() const { return data_; }
    constexpr const char* end() const { return data_ + 4; }
    operator std::basic_string_view<char>() const = delete;
  };

  static_assert(std::is_constructible_v<std::string_view, DeletedConversionOperator>);
  static_assert(std::is_constructible_v<std::string_view, const DeletedConversionOperator>);
  static_assert(std::is_constructible_v<std::string_view, DeletedConstConversionOperator>);
  static_assert(std::is_constructible_v<std::string_view, const DeletedConstConversionOperator>);

  // Test that we're not trying to use the type's conversion operator to string_view in the constructor.
  {
    const DeletedConversionOperator d;
    std::basic_string_view<char> csv = d;
    assert(csv == "test");
  }

  {
    DeletedConstConversionOperator dc;
    std::basic_string_view<char> sv = dc;
    assert(sv == "test");
  }

  return true;
}

static_assert(std::is_constructible_v<std::string_view, std::vector<char>&>);
static_assert(std::is_constructible_v<std::string_view, const std::vector<char>&>);
static_assert(std::is_constructible_v<std::string_view, std::vector<char>&&>);
static_assert(std::is_constructible_v<std::string_view, const std::vector<char>&&>);

using SizedButNotContiguousRange = std::ranges::subrange<random_access_iterator<char*>>;
static_assert(!std::ranges::contiguous_range<SizedButNotContiguousRange>);
static_assert(std::ranges::sized_range<SizedButNotContiguousRange>);
static_assert(!std::is_constructible_v<std::string_view, SizedButNotContiguousRange>);

using ContiguousButNotSizedRange = std::ranges::subrange<contiguous_iterator<char*>, sentinel_wrapper<contiguous_iterator<char*>>, std::ranges::subrange_kind::unsized>;
static_assert(std::ranges::contiguous_range<ContiguousButNotSizedRange>);
static_assert(!std::ranges::sized_range<ContiguousButNotSizedRange>);
static_assert(!std::is_constructible_v<std::string_view, ContiguousButNotSizedRange>);

static_assert(!std::is_constructible_v<std::string_view, std::vector<char16_t>>); // different CharT

struct WithStringViewConversionOperator {
  char* begin() const;
  char* end() const;
  operator std::string_view() const { return {}; }
};

static_assert(std::is_constructible_v<std::string_view, WithStringViewConversionOperator>); // lvalue
static_assert(std::is_constructible_v<std::string_view, const WithStringViewConversionOperator&>); // const lvalue
static_assert(std::is_constructible_v<std::string_view, WithStringViewConversionOperator&&>); // rvalue

template <class CharTraits>
struct WithTraitsType {
  typename CharTraits::char_type* begin() const;
  typename CharTraits::char_type* end() const;
  using traits_type = CharTraits;
};

using CCT = constexpr_char_traits<char>;
static_assert(std::is_constructible_v<std::string_view, WithTraitsType<std::char_traits<char>>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::is_constructible_v<std::wstring_view, WithTraitsType<std::char_traits<wchar_t>>>);
#endif
static_assert(std::is_constructible_v<std::basic_string_view<char, CCT>, WithTraitsType<CCT>>);
static_assert(!std::is_constructible_v<std::string_view, WithTraitsType<CCT>>);  // wrong traits type
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_constructible_v<std::wstring_view, WithTraitsType<std::char_traits<char>>>);  // wrong traits type
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_throwing() {
  struct ThrowingData {
    char* begin() const { return nullptr; }
    char* end() const { return nullptr; }
    char* data() const { throw 42; return nullptr; }
  };
  try {
    ThrowingData x;
    (void) std::string_view(x);
    assert(false);
  } catch (int i) {
    assert(i == 42);
  }

  struct ThrowingSize {
    char* begin() const { return nullptr; }
    char* end() const { return nullptr; }
    size_t size() const { throw 42; return 0; }
  };
  try {
    ThrowingSize x;
    (void) std::string_view(x);
    assert(false);
  } catch (int i) {
    assert(i == 42);
  }
}
#endif

int main(int, char**) {
  test();
  static_assert(test());
#ifndef TEST_HAS_NO_EXCEPTIONS
  test_throwing();
#endif

  return 0;
}

