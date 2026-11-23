//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator,
//          class Hash = hash<iter-key-type<InputIterator>>,
//          class Pred = equal_to<iter-key-type<InputIterator>>,
//          class Allocator = allocator<iter-to-alloc-type<InputIterator>>>
// unordered_multimap(InputIterator, InputIterator, typename see below::size_type = see below,
//                    Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_multimap<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash, Pred,
//                         Allocator>;
//
// template<class Key, class T, class Hash = hash<Key>,
//          class Pred = equal_to<Key>, class Allocator = allocator<pair<const Key, T>>>
// unordered_multimap(initializer_list<pair<Key, T>>,
//                    typename see below::size_type = see below, Hash = Hash(),
//                    Pred = Pred(), Allocator = Allocator())
//   -> unordered_multimap<Key, T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_multimap(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_multimap<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                         hash<iter-key-type<InputIterator>>,
//                         equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_multimap(InputIterator, InputIterator, Allocator)
//   -> unordered_multimap<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                         hash<iter-key-type<InputIterator>>,
//                         equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_multimap(InputIterator, InputIterator, typename see below::size_type, Hash, Allocator)
//   -> unordered_multimap<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash,
//                         equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_multimap(initializer_list<pair<Key, T>>, typename see below::size_type, Allocator)
//   -> unordered_multimap<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_multimap(initializer_list<pair<Key, T>>, Allocator)
//   -> unordered_multimap<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Hash, class Allocator>
// unordered_multimap(initializer_list<pair<Key, T>>, typename see below::size_type, Hash,
//                    Allocator)
//   -> unordered_multimap<Key, T, Hash, equal_to<Key>, Allocator>;
//
// template<ranges::input_range R, class Hash = hash<range-key-type<R>>,
//           class Pred = equal_to<range-key-type<R>>,
//           class Allocator = allocator<range-to-alloc-type<R>>>
//   unordered_multimap(from_range_t, R&&, typename see below::size_type = see below,
//                 Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//     -> unordered_multimap<range-key-type<R>, range-mapped-type<R>, Hash, Pred, Allocator>; // C++23
//
// template<ranges::input_range R, class Allocator>
//   unordered_multimap(from_range_t, R&&, typename see below::size_type, Allocator)
//     -> unordered_multimap<range-key-type<R>, range-mapped-type<R>, hash<range-key-type<R>>,
//                       equal_to<range-key-type<R>>, Allocator>;   // C++23
//
// template<ranges::input_range R, class Allocator>
//   unordered_multimap(from_range_t, R&&, Allocator)
//     -> unordered_multimap<range-key-type<R>, range-mapped-type<R>, hash<range-key-type<R>>,
//                       equal_to<range-key-type<R>>, Allocator>;   // C++23
//
// template<ranges::input_range R, class Hash, class Allocator>
//   unordered_multimap(from_range_t, R&&, typename see below::size_type, Hash, Allocator)
//     -> unordered_multimap<range-key-type<R>, range-mapped-type<R>, Hash,
//                       equal_to<range-key-type<R>>, Allocator>;   // C++23

#include <algorithm> // is_permutation
#include <array>
#include <cassert>
#include <climits> // INT_MAX
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"
#include "type_algorithms.h"

using P  = std::pair<int, long>;
using PC = std::pair<const int, long>;

int main(int, char**) {
  const PC expected_m[] = {{1, 1}, {1, 1}, {2, 2}, {3, 1}, {INT_MAX, 1}};

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr));
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr), 42);
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr), 42, std::hash<short>());
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<int>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr), 42, std::hash<short>(), std::equal_to<>());
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(
        std::begin(arr), std::end(arr), 42, std::hash<short>(), std::equal_to<>(), test_allocator<PC>(0, 41));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 41);
  }

  {
    std::unordered_multimap<int, long> source;
    std::unordered_multimap m(source);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
  }

  {
    std::unordered_multimap<int, long> source;
    std::unordered_multimap m{source}; // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.size() == 0);
  }

  {
    std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>, test_allocator<PC>> source;
    test_allocator<PC> a(0, 42);
    std::unordered_multimap m(source, a);
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.get_allocator().get_id() == 42);
    assert(m.size() == 0);
  }

  {
    std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>, test_allocator<PC>> source;
    test_allocator<PC> a(0, 43);
    std::unordered_multimap m{source, a}; // braces instead of parens
    ASSERT_SAME_TYPE(decltype(m), decltype(source));
    assert(m.get_allocator().get_id() == 43);
    assert(m.size() == 0);
  }

  {
    std::unordered_multimap m{P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}};
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    std::unordered_multimap m({P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}}, 42);
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    std::unordered_multimap m({P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}}, 42, std::hash<short>());
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long, std::hash<short>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    std::unordered_multimap m(
        {P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}}, 42, std::hash<short>(), std::equal_to<>());
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
  }

  {
    std::unordered_multimap m(
        {P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}},
        42,
        std::hash<short>(),
        std::equal_to<>(),
        test_allocator<PC>(0, 44));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 44);
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr), 42, test_allocator<PC>(0, 45));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<int>, std::equal_to<int>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 45);
  }

  {
    const P arr[] = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
    std::unordered_multimap m(std::begin(arr), std::end(arr), 42, std::hash<short>(), test_allocator<PC>(0, 46));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<int>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 46);
  }

  {
    std::unordered_multimap m({P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}}, 42, test_allocator<PC>(0, 47));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<int>, std::equal_to<int>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 47);
  }

  {
    std::unordered_multimap m(
        {P{1, 1L}, P{2, 2L}, P{1, 1L}, P{INT_MAX, 1L}, P{3, 1L}}, 42, std::hash<short>(), test_allocator<PC>(0, 48));
    ASSERT_SAME_TYPE(
        decltype(m), std::unordered_multimap<int, long, std::hash<short>, std::equal_to<int>, test_allocator<PC>>);
    assert(std::is_permutation(m.begin(), m.end(), std::begin(expected_m), std::end(expected_m)));
    assert(m.get_allocator().get_id() == 48);
  }

  {
    // Examples from LWG3025
    std::unordered_multimap m{std::pair{1, 1}, {2, 2}, {3, 3}};
    ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, int>);

    std::unordered_multimap m2{m.begin(), m.end()};
    ASSERT_SAME_TYPE(decltype(m2), std::unordered_multimap<int, int>);
  }

  {
    // Examples from LWG3531
    std::unordered_multimap m1{{std::pair{1, 2}, {3, 4}}, 0};
    ASSERT_SAME_TYPE(decltype(m1), std::unordered_multimap<int, int>);

    using value_type = std::pair<const int, int>;
    std::unordered_multimap m2{{value_type{1, 2}, {3, 4}}, 0};
    ASSERT_SAME_TYPE(decltype(m2), std::unordered_multimap<int, int>);
  }

#if TEST_STD_VER >= 23
  {
    using Range       = std::array<P, 0>;
    using Pred        = test_equal_to<int>;
    using DefaultPred = std::equal_to<int>;
    using Hash        = test_hash<int>;
    using DefaultHash = std::hash<int>;
    using Alloc       = test_allocator<PC>;

    { // (from_range, range)
      std::unordered_multimap c(std::from_range, Range());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long>>);
    }

    { // (from_range, range, n)
      std::unordered_multimap c(std::from_range, Range(), std::size_t());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long>>);
    }

    { // (from_range, range, n, hash)
      std::unordered_multimap c(std::from_range, Range(), std::size_t(), Hash());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, Hash>>);
    }

    { // (from_range, range, n, hash, pred)
      std::unordered_multimap c(std::from_range, Range(), std::size_t(), Hash(), Pred());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, Hash, Pred>>);
    }

    { // (from_range, range, n, hash, pred, alloc)
      std::unordered_multimap c(std::from_range, Range(), std::size_t(), Hash(), Pred(), Alloc());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, Hash, Pred, Alloc>>);
    }

    { // (from_range, range, n, alloc)
      std::unordered_multimap c(std::from_range, Range(), std::size_t(), Alloc());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, DefaultHash, DefaultPred, Alloc>>);
    }

    // TODO(LWG 2713): uncomment this test once the constructor is added.
    { // (from_range, range, alloc)
      //std::unordered_multimap c(std::from_range, Range(), Alloc());
      //static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, DefaultHash, DefaultPred, Alloc>>);
    }

    { // (from_range, range, n, hash, alloc)
      std::unordered_multimap c(std::from_range, Range(), std::size_t(), Hash(), Alloc());
      static_assert(std::is_same_v<decltype(c), std::unordered_multimap<int, long, Hash, DefaultPred, Alloc>>);
    }
  }

  {
    std::vector<std::pair<const int, float>> pair_vec = {{1, 1.1f}, {2, 2.2f}, {3, 3.3f}};
    std::unordered_multimap umm1(pair_vec.begin(), pair_vec.end());
    ASSERT_SAME_TYPE(decltype(umm1), std::unordered_multimap<int, float>);

    std::vector<std::tuple<int, double>> tuple_vec = {{10, 1.1}, {20, 2.2}, {30, 3.3}};
    std::unordered_multimap umm2(tuple_vec.begin(), tuple_vec.end());
    ASSERT_SAME_TYPE(decltype(umm2), std::unordered_multimap<int, double>);

    std::vector<std::array<long, 2>> array_vec = {{100L, 101L}, {200L, 201L}, {300L, 301L}};
    std::unordered_multimap umm3(array_vec.begin(), array_vec.end());
    ASSERT_SAME_TYPE(decltype(umm3), std::unordered_multimap<long, long>);

    std::vector<std::pair<int, char>> non_const_key_pair_vec = {{5, 'a'}, {6, 'b'}};
    std::unordered_multimap umm4(non_const_key_pair_vec.begin(), non_const_key_pair_vec.end());
    ASSERT_SAME_TYPE(decltype(umm4), std::unordered_multimap<int, char>);
  }
#endif

  // [unord.multimap.overview]/5: Use the container's size_type in deduction guides
  {
    using Iter      = const P*;
    using Hash      = std::hash<int>;
    using AllocList = types::type_list<test_allocator<PC>, other_allocator<PC>>;

    // Test multiple allocators with different size_types (unsigned int and size_t)
    types::for_each(
        AllocList(), types::apply_type_identity{[](auto alloc_id) {
          using Alloc    = typename decltype(alloc_id)::type;
          using SizeType = typename std::unordered_multimap<int, long, Hash, std::equal_to<int>, Alloc>::size_type;

          // Ensure deduction guides only accept the exact SizeType and SFINAE away for other integer types.
          types::for_each(
              types::unsigned_integer_types(), types::apply_type_identity{[](auto id) {
                using UI = typename decltype(id)::type;

                // (iter, iter, size, alloc)
                static_assert(SFINAEs_away<std::unordered_multimap, Iter, Iter, OnlyConvertibleTo<UI>, Alloc> ==
                              !std::is_same_v<UI, SizeType>);

                // (iter, iter, size, hash, alloc)
                static_assert(SFINAEs_away<std::unordered_multimap, Iter, Iter, OnlyConvertibleTo<UI>, Hash, Alloc> ==
                              !std::is_same_v<UI, SizeType>);

                // (init_list, size, alloc)
                static_assert(SFINAEs_away_initializer_list<std::unordered_multimap, P, OnlyConvertibleTo<UI>, Alloc> ==
                              !std::is_same_v<UI, SizeType>);

                // (init_list, size, hash, alloc)
                static_assert(
                    SFINAEs_away_initializer_list<std::unordered_multimap, P, OnlyConvertibleTo<UI>, Hash, Alloc> ==
                    !std::is_same_v<UI, SizeType>);

#if TEST_STD_VER >= 23
                using Range = std::array<P, 0>;

                // (from_range, range, size, alloc)
                static_assert(
                    SFINAEs_away<std::unordered_multimap, std::from_range_t, Range, OnlyConvertibleTo<UI>, Alloc> ==
                    !std::is_same_v<UI, SizeType>);

                // (from_range, range, size, hash, alloc)
                static_assert(
                    SFINAEs_away<std::unordered_multimap,
                                 std::from_range_t,
                                 Range,
                                 OnlyConvertibleTo<UI>,
                                 Hash,
                                 Alloc> == !std::is_same_v<UI, SizeType>);
#endif
                // (iter, iter, size, alloc) - runtime check
                if constexpr (std::is_same_v<UI, SizeType>) {
                  const P arr[]       = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 1}, {3, 1}};
                  const PC expected[] = {{1, 1}, {1, 1}, {2, 2}, {3, 1}, {INT_MAX, 1}};
                  std::unordered_multimap m(std::begin(arr), std::end(arr), OnlyConvertibleTo<SizeType>{}, Alloc());
                  ASSERT_SAME_TYPE(decltype(m), std::unordered_multimap<int, long, Hash, std::equal_to<int>, Alloc>);
                  assert(std::is_permutation(m.begin(), m.end(), std::begin(expected), std::end(expected)));
                }
              }});
        }});
  }

  UnorderedContainerDeductionGuidesSfinaeAway<std::unordered_multimap, std::unordered_multimap<int, long>>();

  return 0;
}
