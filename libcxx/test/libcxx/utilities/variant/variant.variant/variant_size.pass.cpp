//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"

template <class Sequence>
struct make_variant_imp;

template <std::size_t ...Indices>
struct make_variant_imp<std::integer_sequence<std::size_t, Indices...>> {
  template <std::size_t> using AlwaysChar = char;
  using type = std::variant<AlwaysChar<Indices>...>;
};

template <std::size_t N>
using make_variant_t = typename make_variant_imp<std::make_index_sequence<N>>::type;

constexpr bool ExpectEqual =
#ifdef _LIBCPP_ABI_VARIANT_INDEX_TYPE_OPTIMIZATION
  false;
#else
  true;
#endif

template <class IndexType>
void test_index_type() {
  using Lim = std::numeric_limits<IndexType>;
  using T1 = make_variant_t<Lim::max() - 1>;
  using T2 = make_variant_t<Lim::max()>;
  static_assert((sizeof(T1) == sizeof(T2)) == ExpectEqual, "");
}

template <class IndexType>
void test_index_internals() {
  using Lim = std::numeric_limits<IndexType>;
  static_assert(std::__choose_index_type(Lim::max() -1) !=
                std::__choose_index_type(Lim::max()), "");
  static_assert(std::is_same_v<
      std::__variant_index_t<Lim::max()-1>,
      std::__variant_index_t<Lim::max()>
    > == ExpectEqual, "");
  using IndexT = std::__variant_index_t<Lim::max()-1>;
  using IndexLim = std::numeric_limits<IndexT>;
  static_assert(std::__variant_npos<IndexT> == IndexLim::max(), "");
}

template <class LargestType>
struct type_with_index {
  LargestType largest;
#ifdef _LIBCPP_ABI_VARIANT_INDEX_TYPE_OPTIMIZATION
  unsigned char index;
#else
  unsigned int index;
#endif
};

int main(int, char**) {
  test_index_type<unsigned char>();
  // This won't compile due to template depth issues.
  //test_index_type<unsigned short>();
  test_index_internals<unsigned char>();
  test_index_internals<unsigned short>();

  // Test that std::variant achieves the expected size. See https://llvm.org/PR61095.
  static_assert(sizeof(std::variant<char, char, char>) == sizeof(type_with_index<char>));
  static_assert(sizeof(std::variant<int, int, int>) == sizeof(type_with_index<int>));
  static_assert(sizeof(std::variant<long, long, long>) == sizeof(type_with_index<long>));
  static_assert(sizeof(std::variant<char, int, long>) == sizeof(type_with_index<long>));
  static_assert(sizeof(std::variant<std::size_t, std::size_t, std::size_t>) == sizeof(type_with_index<std::size_t>));

  return 0;
}
